import torch
import numpy as np
import torch.nn.functional as F
from torch import Tensor, nn
from typing import List, Dict, Tuple
from ml4co_kit import TSPTask, to_numpy, to_tensor
from ml4co.fast_t2t.tsp.tsp_env import TSPEnv
from ml4co.fast_t2t.tsp.module import TSPModel
from ml4co.fast_t2t.tsp.tsp_diffusion import TSPDiffusion
from ml4co.fast_t2t.common import MetaPLModel, MetaDataBatch, InferenceSchedule
from ml4co.fast_t2t.tsp.lib import c_tsp_greedy, c_tsp_2opt, pytorch_tsp_gpu_2opt


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
        gpu_2opt: bool = False,
    ):
        # Super Args
        super(TSPPLModel, self).__init__(
            # Basic Args
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
        self.gpu_2opt = gpu_2opt

        # Hint
        self.env: TSPEnv
        self.model: TSPModel

    def _train_step(
        self, 
        batch_task_data: List[TSPTask], 
        batch_processed_data: MetaDataBatch
    ) -> Dict[str, float]:
        # Get data
        if self.env.device == "cuda":
            batch_processed_data.to_cuda()
        points = batch_processed_data.node_feature.float()
        edge_index = batch_processed_data.edge_index
        gt = batch_processed_data.ground_truth.float()

        # Consistency Sample
        t1: Tensor = torch.randint(
            low=1, high=1001, size=(1,), device=self.env.device
        )
        t2 = (self.cm_alpha * t1).int()
        s1, s2 = self.diffusion.sample(x=gt, t1=t1, t2=t2)

        # Optional for robustness
        s1 = (s1 * 2 - 1) * (1.0 + 0.05 * torch.rand_like(s1)) 
        s2 = (s2 * 2 - 1) * (1.0 + 0.05 * torch.rand_like(s2))
        
        # Forward
        logits_1 = self.model.forward(points, s1, t1, edge_index)
        logits_2 = self.model.forward(points, s2, t2, edge_index)

        # Calculate loss
        loss_func = nn.CrossEntropyLoss()
        loss_1 = loss_func(logits_1, gt.long())
        loss_2 = loss_func(logits_2, gt.long())
        loss = loss_1 + loss_2

        # Return metrics
        return {"loss": loss}

    def _val_step(
        self, 
        batch_task_data: List[TSPTask], 
        batch_processed_data: MetaDataBatch
    ) -> Dict[str, float]:
        # Get data
        if self.env.device == "cuda":
            batch_processed_data.to_cuda()

        # Evaluate (Is = 1)
        avg_gap_1, avg_match_1, loss_1 = self._val_step_core(
            batch_task_data=batch_task_data,
            batch_processed_data=batch_processed_data,
            inference_steps=1
        )

        # Evaluate (Is = 5)
        avg_gap_5, avg_match_5, loss_5 = self._val_step_core(
            batch_task_data=batch_task_data,
            batch_processed_data=batch_processed_data,
            inference_steps=5
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
        inference_steps: int
    ) -> Tuple[float, float, Tensor]:
        # Get data
        points = batch_processed_data.node_feature.float()
        bs = len(batch_task_data)
        ptr = batch_processed_data.ptr
        nodes_num = batch_task_data[0].nodes_num
        gt = batch_processed_data.ground_truth

        # Get edge index
        edge_index = batch_processed_data.edge_index
        _minus = ptr[:-1][:, None, None].to(self.env.device)
        th_edge_index = edge_index.reshape(2, bs, -1).transpose(0, 1)
        th_edge_index = (th_edge_index - _minus).reshape(bs, 2, -1)
        ed_0 = th_edge_index[:, 0, :]
        ed_1 = th_edge_index[:, 1, :]
        ed_for_greedy = ed_0 * nodes_num + ed_1

        # Prepare for inference
        time_schedule = InferenceSchedule("cosine", 1000, inference_steps)
        sols_array = torch.randn(bs*nodes_num*self.knn)
        st = (sols_array  > 0).float().to(self.env.device)

        # Inference
        for idx in range(inference_steps):
            # Time
            t1, t2 = time_schedule(idx)
            t1 = torch.tensor([t1]).to(self.env.device)

            # Optional for robustness
            st = (st * 2 - 1) * (1.0 + 0.05 * torch.rand_like(st))

            # Forward
            logits = self.model.forward(points, st, t1, edge_index)

            # Get next st
            if t2 != 0:
                heatmap = F.softmax(logits, dim=-1)[:, 1]
                pred_ber = torch.bernoulli(heatmap.clamp(0, 1))
                pred_ber_onehot: Tensor = F.one_hot(pred_ber.long(), num_classes=2)
                Q_bar = torch.from_numpy(self.diffusion.Q_bar[t2]).float().to(self.device)
                prob = torch.matmul(pred_ber_onehot.float(), Q_bar)
                st = torch.bernoulli(prob[..., 1].clamp(0, 1))

        # Loss
        loss_func = nn.CrossEntropyLoss()
        loss = loss_func(logits, gt.long())

        # Get heatmap
        heatmap = F.softmax(logits, dim=-1)[:, 1]

        # Match
        avg_match = heatmap[gt == 1].mean()
        
        # Greedy Decode
        heatmap = heatmap.reshape(bs, -1)
        top_idx = torch.topk(heatmap, k=20*nodes_num, dim=-1)[1]
        top_edges = ed_for_greedy.gather(dim=-1, index=top_idx)
        top_edges = to_numpy(top_edges)
        greedy_sols = c_tsp_greedy(
            top_edges=top_edges, nodes_num=nodes_num, num_workers=bs
        )

        # Evaluate
        gaps = list()
        for td, _greedy_sol in zip(batch_task_data, greedy_sols):
            td.from_data(sol=_greedy_sol, ref=False)
            gap = td.evaluate_w_gap()[2]
            gaps.append(gap)
        avg_gap = np.mean(gaps)

        # Return
        return avg_gap, avg_match, loss

    def _solve_step(
        self, 
        batch_task_data: List[TSPTask], 
        batch_processed_data: MetaDataBatch,
        solve_runs: int = 1
    ) -> Dict[str, float]:

        ################################
        #      1. Preprocess data      # 
        ################################

        # 1.1 Get data
        if self.env.device == "cuda":
            batch_processed_data.to_cuda()
        points = batch_processed_data.node_feature.float()
        bs = len(batch_task_data)
        ptr = batch_processed_data.ptr
        pbs = len(ptr) - 1
        nodes_num = batch_task_data[0].nodes_num
        np_points = to_numpy(points).reshape(pbs, nodes_num, 2)

        # 1.2 Get edge index
        edge_index = batch_processed_data.edge_index
        _minus = ptr[:-1][:, None, None].to(self.env.device)
        th_edge_index = edge_index.reshape(2, pbs, -1).transpose(0, 1)
        th_edge_index = (th_edge_index - _minus).reshape(pbs, 2, -1)

        # 1.3 For greedy decode
        ed_0 = th_edge_index[:, 0, :]
        ed_1 = th_edge_index[:, 1, :]
        ed_for_greedy = ed_0 * nodes_num + ed_1

        ################################
        #    2. Initialize solution    # 
        ################################

        # 2.1 Gaussian random solution
        sol_adj = torch.randn(pbs*nodes_num*self.knn)
        st = (sol_adj  > 0).float().to(self.env.device)
        
        # 2.2 Prepare for inference
        time_schedule = InferenceSchedule("cosine", 1000, self.cm_steps)

        # 2.3 Inference
        for idx in range(self.cm_steps):
            # Time
            t1, t2 = time_schedule(idx)
            t1 = torch.tensor([t1]).to(self.env.device)
            
            # Optional for robustness
            st = (st * 2 - 1) * (1.0 + 0.05 * torch.rand_like(st))

            # Multi-step inference
            if t2 != 0:
                logits = self.model.forward(points, st, t1, edge_index)
                heatmap = F.softmax(logits, dim=-1)[:, 1]
                pred_ber = torch.bernoulli(heatmap.clamp(0, 1))
                pred_ber_onehot: Tensor = F.one_hot(pred_ber.long(), num_classes=2)
                Q_bar = torch.from_numpy(self.diffusion.Q_bar[t2]).float().to(self.device)
                prob = torch.matmul(pred_ber_onehot.float(), Q_bar)
                st = torch.bernoulli(prob[..., 1].clamp(0, 1))
            else:
                logits = self.model.forward(points, st, t1, edge_index)
                heatmap = F.softmax(logits, dim=-1)[:, 1]

        # 2.4 Geedy decode
        heatmap = heatmap.reshape(pbs, -1)
        top_idx = torch.topk(heatmap, k=20*nodes_num, dim=-1)[1]
        top_edges = ed_for_greedy.gather(dim=-1, index=top_idx)
        top_edges = to_numpy(top_edges)
        np_greedy_sols = c_tsp_greedy(
            top_edges=top_edges, nodes_num=nodes_num, num_workers=pbs
        )

        # 2.5 Local search
        if self.gpu_2opt:
            th_greedy_sols = to_tensor(np_greedy_sols).to("cuda")
            points_pbs_cuda = points.reshape(pbs, nodes_num, 2)
            best_sols = pytorch_tsp_gpu_2opt(points=points_pbs_cuda, tours=th_greedy_sols)
            best_sols = to_numpy(best_sols)
        else:
            best_sols = c_tsp_2opt(
                points=np_points, tours=np_greedy_sols, num_workers=pbs
            )
        
        # 2.7 Get costs
        costs = self._get_cur_costs(
            batch_task_data, best_sols, solve_runs
        )
        costs = costs.reshape(bs, solve_runs)
        best_idx = np.argmin(costs, axis=1)
        best_idx = solve_runs * np.arange(bs) + best_idx
        best_sols = best_sols[best_idx, :]            

        # 2.8 Store Best Solution
        for td, best_sol in zip(batch_task_data, best_sols):
            td.from_data(sol=best_sol, ref=False)
        return

    def _get_cur_costs(
        self, 
        batch_task_data: List[TSPTask],
        cur_sols: np.ndarray,
        solve_runs: int
    ) -> np.ndarray:
        costs = list()
        for idx, cur_sol in enumerate(cur_sols):
            td = batch_task_data[idx // solve_runs]
            costs.append(td.evaluate(cur_sol))
        return np.array(costs)