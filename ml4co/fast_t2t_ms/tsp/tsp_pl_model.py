
import numpy as np
import mindspore as ms
from ml4co_kit import TSPTask
from typing import List, Dict, Tuple
from mindspore import Tensor, nn, ops
from ml4co.fast_t2t_ms.tsp.tsp_env import TSPEnv
from ml4co.fast_t2t_ms.tsp.module import TSPModel
from ml4co.fast_t2t_ms.tsp.tsp_diffusion import TSPDiffusion
from ml4co.ms_utils import to_numpy, to_tensor
from ml4co.fast_t2t_ms.common import MetaPLModel, MetaDataBatch, InferenceSchedule
from ml4co.fast_t2t_ms.tsp.lib import c_tsp_greedy, c_tsp_2opt, mindspore_tsp_2opt


class TSPPLModel(MetaPLModel):
    def __init__(
        self,
        # Basic Args
        env: TSPEnv,
        model: TSPModel,
        lr_scheduler: str = "cosine-decay",
        learning_rate: float = 2e-4,
        weight_decay: float = 1e-4,
        weight_path: str = None,
        cm_alpha: float = 0.2,
        cm_steps: int = 5,
        knn: int = 50,
        ms_2opt: bool = False,
    ):
        # Super Args
        super().__init__(
            env=env,
            model=model,
            lr_scheduler=lr_scheduler,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            weight_path=weight_path,
            cm_alpha=cm_alpha,
            cm_steps=cm_steps,
        )

        # Diffusion Args
        self.diffusion = TSPDiffusion(T=1000)

        # TSP Args
        self.knn = knn
        self.ms_2opt = ms_2opt

        # Hint
        self.env: TSPEnv
        self.model: TSPModel
        self._ce_loss = nn.CrossEntropyLoss()

    def _maybe_to_device(self, batch_processed_data: MetaDataBatch):
        # Process context + tensor storage must both match env.device.
        self._sync_device()
        batch_processed_data.to_device(self.env.device)

    def _train_step(
        self,
        batch_task_data: List[TSPTask],
        batch_processed_data: MetaDataBatch,
    ) -> Dict[str, Tensor]:
        # Get data
        self._maybe_to_device(batch_processed_data)
        points = batch_processed_data.node_feature.astype(ms.float32)
        edge_index = batch_processed_data.edge_index
        gt = batch_processed_data.ground_truth.astype(ms.float32)

        # Consistency Sample
        t1 = ops.randint(1, 1001, (1,), dtype=ms.int32)
        t2 = ops.cast(self.cm_alpha * t1.astype(ms.float32), ms.int32)
        s1, s2 = self.diffusion.sample(x=gt, t1=t1, t2=t2)

        # Optional for robustness
        _u0, _u1 = ms.Tensor(0.0, ms.float32), ms.Tensor(1.0, ms.float32)
        s1 = (s1 * 2 - 1) * (1.0 + 0.05 * ops.uniform(s1.shape, _u0, _u1))
        s2 = (s2 * 2 - 1) * (1.0 + 0.05 * ops.uniform(s2.shape, _u0, _u1))

        # Forward
        logits_1 = self.model(points, s1, t1, edge_index)
        logits_2 = self.model(points, s2, t2, edge_index)

        # Calculate loss
        loss_1 = self._ce_loss(logits_1, gt.astype(ms.int32))
        loss_2 = self._ce_loss(logits_2, gt.astype(ms.int32))
        loss = loss_1 + loss_2

        # Return metrics
        return {"loss": loss}

    def _val_step(
        self,
        batch_task_data: List[TSPTask],
        batch_processed_data: MetaDataBatch,
    ) -> Dict[str, float]:
        # Get data
        self._maybe_to_device(batch_processed_data)

        # Evaluate (Is = 1)
        avg_gap_1, avg_match_1, loss_1 = self._val_step_core(
            batch_task_data, batch_processed_data, inference_steps=1
        )

        # Evaluate (Is = 5)
        avg_gap_5, avg_match_5, loss_5 = self._val_step_core(
            batch_task_data, batch_processed_data, inference_steps=5
        )

        # Merge
        loss = loss_1 + loss_5

        # Return metrics
        return {
            "loss": loss,
            "ag_1": avg_gap_1,
            "ag_5": avg_gap_5,
            "am_1": avg_match_1,
            "am_5": avg_match_5,
        }

    def _val_step_core(
        self,
        batch_task_data: List[TSPTask],
        batch_processed_data: MetaDataBatch,
        inference_steps: int,
    ) -> Tuple[float, float, Tensor]:
        # Get data
        points = batch_processed_data.node_feature.astype(ms.float32)
        bs = len(batch_task_data)
        ptr = batch_processed_data.ptr
        nodes_num = batch_task_data[0].nodes_num
        gt = batch_processed_data.ground_truth

        # Get edge index (per-graph, for greedy flat indices)
        edge_index = batch_processed_data.edge_index
        _minus = ops.expand_dims(ops.expand_dims(ptr[:-1], -1), -1)
        th_edge_index = edge_index.reshape(2, bs, -1)
        th_edge_index = ops.transpose(th_edge_index, (1, 0, 2))
        th_edge_index = (th_edge_index - _minus).reshape(bs, 2, -1)
        ed_0 = th_edge_index[:, 0, :]
        ed_1 = th_edge_index[:, 1, :]
        ed_for_greedy = ed_0 * nodes_num + ed_1

        # Prepare for inference
        time_schedule = InferenceSchedule("cosine", 1000, inference_steps)
        sols_array = ops.standard_normal((bs * nodes_num * self.knn,))
        st = (sols_array > 0).astype(ms.float32)

        # Inference
        logits = None
        for idx in range(inference_steps):
            # Time
            t1, t2 = time_schedule(idx)
            t1_t = ms.Tensor([t1], ms.int32)

            # Optional for robustness
            _u0, _u1 = ms.Tensor(0.0, ms.float32), ms.Tensor(1.0, ms.float32)
            st = (st * 2 - 1) * (1.0 + 0.05 * ops.uniform(st.shape, _u0, _u1))

            # Forward
            logits = self.model(points, st, t1_t, edge_index)

            # Get next st
            if t2 != 0:
                heatmap = nn.Softmax(axis=-1)(logits)[:, 1]
                pred_ber = ops.bernoulli(ops.clip_by_value(heatmap, 0.0, 1.0))
                pred_ber_onehot = ops.one_hot(
                    pred_ber.astype(ms.int32),
                    2,
                    ms.Tensor(1.0, ms.float32),
                    ms.Tensor(0.0, ms.float32),
                )
                Q_bar = ms.Tensor(self.diffusion.Q_bar[t2], ms.float32)
                prob = ops.matmul(pred_ber_onehot.astype(ms.float32), Q_bar)
                st = ops.bernoulli(ops.clip_by_value(prob[..., 1], 0.0, 1.0))

        # Loss
        loss = self._ce_loss(logits, gt.astype(ms.int32))

        # Get heatmap
        heatmap = nn.Softmax(axis=-1)(logits)[:, 1]

        # Match
        avg_match = float(heatmap[gt == 1].mean().asnumpy())

        # Greedy Decode
        heatmap_2d = heatmap.reshape(bs, -1)
        topk = ops.TopK(sorted=True)
        _, top_idx = topk(heatmap_2d, 20 * nodes_num)
        top_edges = ops.gather_elements(ed_for_greedy, -1, top_idx)
        top_edges = to_numpy(top_edges).astype(np.int32)
        greedy_sols = c_tsp_greedy(
            top_edges=top_edges, nodes_num=nodes_num, num_workers=bs
        )

        # Evaluate
        gaps = []
        for td, _greedy_sol in zip(batch_task_data, greedy_sols):
            td.from_data(sol=_greedy_sol, ref=False)
            gaps.append(td.evaluate_w_gap()[2])
        avg_gap = float(np.mean(gaps))

        # Return
        return avg_gap, avg_match, loss

    def _solve_step(
        self,
        batch_task_data: List[TSPTask],
        batch_processed_data: MetaDataBatch,
        solve_runs: int = 1,
    ):
        ################################
        #      1. Preprocess data      #
        ################################

        # 1.1 Get data
        self._maybe_to_device(batch_processed_data)
        points = batch_processed_data.node_feature.astype(ms.float32)
        bs = len(batch_task_data)
        ptr = batch_processed_data.ptr
        pbs = int(ptr.shape[0]) - 1
        nodes_num = batch_task_data[0].nodes_num
        np_points = to_numpy(points).reshape(pbs, nodes_num, 2)

        # 1.2 Get edge index
        edge_index = batch_processed_data.edge_index
        _minus = ops.expand_dims(ops.expand_dims(ptr[:-1], -1), -1)
        th_edge_index = edge_index.reshape(2, pbs, -1)
        th_edge_index = ops.transpose(th_edge_index, (1, 0, 2))
        th_edge_index = (th_edge_index - _minus).reshape(pbs, 2, -1)

        # 1.3 For greedy decode
        ed_0 = th_edge_index[:, 0, :]
        ed_1 = th_edge_index[:, 1, :]
        ed_for_greedy = ed_0 * nodes_num + ed_1

        ################################
        #    2. Initialize solution    #
        ################################

        # 2.1 Gaussian random solution
        sol_adj = ops.standard_normal((pbs * nodes_num * self.knn,))
        st = (sol_adj > 0).astype(ms.float32)

        # 2.2 Prepare for inference
        time_schedule = InferenceSchedule("cosine", 1000, self.cm_steps)

        # 2.3 Inference
        heatmap = None
        for idx in range(self.cm_steps):
            # Time
            t1, t2 = time_schedule(idx)
            t1_t = ms.Tensor([t1], ms.int32)

            # Optional for robustness
            _u0, _u1 = ms.Tensor(0.0, ms.float32), ms.Tensor(1.0, ms.float32)
            st = (st * 2 - 1) * (1.0 + 0.05 * ops.uniform(st.shape, _u0, _u1))

            # Multi-step inference
            logits = self.model(points, st, t1_t, edge_index)
            heatmap = nn.Softmax(axis=-1)(logits)[:, 1]
            if t2 != 0:
                pred_ber = ops.bernoulli(ops.clip_by_value(heatmap, 0.0, 1.0))
                pred_ber_onehot = ops.one_hot(
                    pred_ber.astype(ms.int32),
                    2,
                    ms.Tensor(1.0, ms.float32),
                    ms.Tensor(0.0, ms.float32),
                )
                Q_bar = ms.Tensor(self.diffusion.Q_bar[t2], ms.float32)
                prob = ops.matmul(pred_ber_onehot.astype(ms.float32), Q_bar)
                st = ops.bernoulli(ops.clip_by_value(prob[..., 1], 0.0, 1.0))

        # 2.4 Greedy decode
        heatmap = heatmap.reshape(pbs, -1)
        topk = ops.TopK(sorted=True)
        _, top_idx = topk(heatmap, 20 * nodes_num)
        top_edges = ops.gather_elements(ed_for_greedy, -1, top_idx)
        top_edges = to_numpy(top_edges).astype(np.int32)
        np_greedy_sols = c_tsp_greedy(
            top_edges=top_edges, nodes_num=nodes_num, num_workers=pbs
        )

        # 2.5 Local search
        if self.ms_2opt:
            th_greedy_sols = to_tensor(np_greedy_sols).astype(ms.int32)
            points_pbs = points.reshape(pbs, nodes_num, 2)
            best_sols = mindspore_tsp_2opt(
                points=points_pbs, tours=th_greedy_sols
            )
            best_sols = to_numpy(best_sols)
        else:
            best_sols = c_tsp_2opt(
                points=np_points, tours=np_greedy_sols, num_workers=pbs
            )

        # 2.6 Get costs and pick best run
        costs = self._get_cur_costs(batch_task_data, best_sols, solve_runs)
        costs = costs.reshape(bs, solve_runs)
        best_idx = np.argmin(costs, axis=1)
        best_idx = solve_runs * np.arange(bs) + best_idx
        best_sols = best_sols[best_idx, :]

        # 2.7 Store Best Solution
        for td, best_sol in zip(batch_task_data, best_sols):
            td.from_data(sol=best_sol, ref=False)

    def _get_cur_costs(
        self,
        batch_task_data: List[TSPTask],
        cur_sols: np.ndarray,
        solve_runs: int,
    ) -> np.ndarray:
        costs = []
        for idx, cur_sol in enumerate(cur_sols):
            td = batch_task_data[idx // solve_runs]
            costs.append(td.evaluate(cur_sol))
        return np.array(costs)
