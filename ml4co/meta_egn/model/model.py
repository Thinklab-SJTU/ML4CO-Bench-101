import torch
import numpy as np
from tqdm import trange
from ml4co_kit import to_numpy
from meta_egn.env.env import MetaEGNEnv
from meta_egn.model.decoder import MetaEGNDecoder
from ml4co.meta_egn.model.encoder import GNNEncoder, MetaLearner, get_diracs
from torch.nn.utils.convert_parameters import vector_to_parameters, parameters_to_vector


class MetaEGNModel(object):
    def __init__(
        self,
        env: MetaEGNEnv,
        encoder: GNNEncoder,
        decoder: MetaEGNDecoder,
        train_outer_steps: int = 10000,
        train_inner_steps: int = 1,
        train_inner_samples: int = 96,
        val_inner_steps: int = 0,
        inner_lr: float = 5e-5,
        outer_lr: float = 1e-3,
        scalar_lam: bool = True,
        lam: float = 1.0, # regularization in inner steps
        penalty_coeff: float = 0.5,
        save_n_epochs: int = 10,
        save_path: str = "path/to/save/weights",
        weight_path: str = None
    ):
        self.env: MetaEGNEnv = env
        self.model: GNNEncoder = encoder
        self.decoder: MetaEGNDecoder = decoder
        self.train_outer_steps =  train_outer_steps
        self.train_inner_steps = train_inner_steps
        self.train_inner_samples = train_inner_samples
        self.val_inner_steps = val_inner_steps
        self.scalar_lam = scalar_lam
        self.lam = lam
        self.save_n_epochs = save_n_epochs
        self.save_path = save_path
        self.penalty_coeff = penalty_coeff
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr

        self.meta_learner = MetaLearner(
            model=self.model,
            loss_function=torch.nn.CrossEntropyLoss(), 
            inner_lr=inner_lr,
            outer_lr=outer_lr,
            device=self.env.device,
            task=env.task
        )
        
        # load pretrained weights if needed
        if weight_path is not None:
            self.state_dict = torch.load(weight_path, map_location=self.env.device)
            self.model.load_state_dict(self.state_dict, strict=True)
        self.model.to(self.env.device)

    def model_train(self):
        self.model.train()
        train_dataloader = self.env.train_dataloader()
        val_dataloader = self.env.val_dataloader()

        meta_learner = self.meta_learner
            
        init_params = meta_learner.get_params()
        device = self.env.device
        lam = torch.tensor(self.lam) if self.scalar_lam else torch.ones(init_params.shape[0]) * self.lam
        lam = lam.to(self.env.device)

        num_tasks = len(train_dataloader)
        
        best_val_obj = 0 if self.env.task != "MVC" else 1e6
        penalty_coeff = self.penalty_coeff

        tbar = trange(self.train_outer_steps)
        for outstep in tbar:
            if outstep > 10 and outstep % 800 == 0 and self.env.task == "MCl":
                penalty_coeff = penalty_coeff + 0.5
            task_mb = np.random.choice(num_tasks, size=self.train_inner_samples)
            for k in range(self.train_inner_steps):
                old_parameters = parameters_to_vector(meta_learner.model.parameters())
                losses_q = torch.tensor([0.0]).to(self.env.device)
                data_index = 0
                for idx in task_mb:
                    task = self.env.train_dataset[idx] # get task
                    task = task.to(self.env.device)
                    
                    tl_before = meta_learner.get_loss(
                        x=task['x'], 
                        edge_index=task['edge_index'], 
                        batch=task['train_batch'], 
                        penalty_coefficient=penalty_coeff, 
                        device=device
                    )
                    
                    new_grad = torch.autograd.grad(
                        only_inputs=tl_before, 
                        inputs=meta_learner.model.parameters(), 
                        retain_graph=True, 
                        create_graph=True
                    )
                    
                    new_params = parameters_to_vector(meta_learner.model.parameters()) - \
                        self.inner_lr * parameters_to_vector(new_grad)
                    vector_to_parameters(new_params, meta_learner.model.parameters())
                    
                    tl_after = meta_learner.get_loss(
                        x=task['x'], 
                        edge_index=task['edge_index'], 
                        batch=task['train_batch'], 
                        penalty_coefficient=penalty_coeff, 
                        device=device
                    )
                    tl_after = tl_after.reshape(-1, 1)

                    if data_index == 0:
                        losses_q = tl_after
                    else:
                        losses_q = torch.cat((losses_q, tl_after), 0)
                    vector_to_parameters(old_parameters, meta_learner.model.parameters())
                    
                    data_index = data_index + 1

                loss_q = torch.mean(losses_q)
                meta_learner.outer_opt.zero_grad()
                loss_q.backward()
                meta_learner.outer_opt.step()
                tbar.set_description(f"Training loss: {loss_q.item()}")

            if outstep > 0 and outstep % (5 * self.save_n_epochs) == 0:
                torch.save(meta_learner.model.state_dict(), self.save_path.format(outstep, ""))
            if outstep == self.train_outer_steps - 1:
                torch.save(meta_learner.model.state_dict(), self.save_path.format("last", ""))

            # validation
            if outstep % self.save_n_epochs == 0:
                print("validating...")
                model_output = np.zeros(len(val_dataloader))
                if self.env.task == "MVC":
                    model_output += 1e6
                model_index = -1
                for data in val_dataloader:
                    model_index = model_index + 1
                    for k in range(8):
                        data_prime = get_diracs(
                            data=data.to(self.env.device), 
                            N=1, 
                            effective_volume_range=0.15, 
                            receptive_field=self.model.receptive_field
                        )
                        data_prime = data_prime.to(self.env.device)
                        probs = meta_learner.model.forward(
                            data_prime.x, data_prime.edge_index, data_prime.batch, None
                        )
                        heatmap = probs.squeeze(-1)
                        sol, obj = self.decode(
                            heatmap=heatmap, 
                            num_nodes=data_prime.x.shape[0], 
                            edge_index=data_prime.edge_index
                        )
                        if (self.env.task in ["MCl", "MIS", "MCut"] and obj.item() > model_output[model_index]) or \
                            (self.env.task in ["MVC"] and obj.item() < model_output[model_index]):
                                model_output[model_index] = obj
                objs = [model_output[i] for i in range(len(model_output))]
                val_obj = np.array(objs).mean()
                if (self.env.task in ["MCl", "MIS", "MCut"] and val_obj > best_val_obj) or \
                    (self.env.task in ["MVC"] and val_obj < best_val_obj):
                    best_val_obj = val_obj
                    torch.save(meta_learner.model.state_dict(), self.save_path.format(outstep, best_val_obj))
                tbar.set_description(f"validation cost: {val_obj:.3f}, best: {best_val_obj:.3f}")

    def model_inference(self, data):
        best_obj = 0 if self.env.task != "MVC" else 1e6
        for k in range(8):
            data_prime = get_diracs(
                data=data.to(self.env.device), 
                N=1, 
                device=self.env.device, 
                effective_volume_range=0.15, 
                receptive_field=self.model.receptive_field
            )
            data_prime = data_prime.to(self.env.device)
            self.model.reset_parameters()
            self.model.load_state_dict(self.state_dict)
            probs = self.model(data_prime.x, data_prime.edge_index, data_prime.batch, None)
            sol, obj = self.decode(heatmap=probs.squeeze(), num_nodes=len(data_prime.x),
                                    edge_index=data_prime.edge_index, egn_rlsa_flag=(k==7))
            if (self.env.task in ["MCl", "MIS", "MCut"] and obj.item() > best_obj) or \
                (self.env.task in ["MVC"] and obj.item() < best_obj):
                best_obj = obj.item()
                final_sol = sol
        return to_numpy(final_sol)

    def decode(self, heatmap, num_nodes, edge_index, egn_rlsa_flag=False):
        return self.decoder.decode(
            heatmap=heatmap, 
            task=self.env.task, 
            nodes_num=num_nodes, 
            edge_index=edge_index, 
            egn_rlsa_flag=egn_rlsa_flag
        )