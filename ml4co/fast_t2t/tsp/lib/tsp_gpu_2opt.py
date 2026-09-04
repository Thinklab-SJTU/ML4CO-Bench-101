import torch
from torch import Tensor


def pytorch_tsp_gpu_2opt(
    init_tours: Tensor,
    points: Tensor,
    max_iters: int = 5000,
    eps: float = 1e-6,
) -> Tensor:
    """
    Two-opt local search (matches ML4CO-Kit ``_torch_tsp_2opt_ls``).

    Args:
        init_tours: (B, V+1) int (closed tour indices).
        points: (V, 2) or (B, V, 2) float.
        max_iters: max outer iterations.
        device: where to run (should match tensors for zero-copy).
        eps: improvement threshold.
    """
    # Preparation
    device = points.device
    tours = init_tours.to(device)
    batch_size = tours.shape[0]
    num_nodes = tours.shape[1] - 1

    if points.dim() == 2:
        points = points.unsqueeze(0).expand(batch_size, -1, -1)
    elif points.dim() == 3:
        if points.shape[0] != batch_size:
            raise ValueError(
                f"Batch size mismatch: tours {batch_size}, points {points.shape[0]}"
            )
        if points.shape[1] != num_nodes:
            raise ValueError(
                f"Node count mismatch: tours imply V={num_nodes}, points {points.shape[1]}"
            )
    else:
        raise ValueError(
            f"Invalid points shape {points.shape}, expected (V, 2) or (B, V, 2)"
        )

    tours = tours.long()

    # Local search
    with torch.inference_mode():
        iterator = 0
        while True:
            batch_indices = (
                torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, num_nodes)
            )
            tour_points = points[batch_indices, tours[:, :-1]]
            tour_points_next = points[batch_indices, tours[:, 1:]]

            points_i = tour_points.unsqueeze(2)
            points_j = tour_points.unsqueeze(1)
            points_i_plus_1 = tour_points_next.unsqueeze(2)
            points_j_plus_1 = tour_points_next.unsqueeze(1)

            A_ij = torch.sqrt(torch.sum((points_i - points_j) ** 2, dim=-1))
            A_i_plus_1_j_plus_1 = torch.sqrt(
                torch.sum((points_i_plus_1 - points_j_plus_1) ** 2, dim=-1)
            )
            A_i_i_plus_1 = torch.sqrt(
                torch.sum((points_i - points_i_plus_1) ** 2, dim=-1)
            )
            A_j_j_plus_1 = torch.sqrt(
                torch.sum((points_j - points_j_plus_1) ** 2, dim=-1)
            )

            change = A_ij + A_i_plus_1_j_plus_1 - A_i_i_plus_1 - A_j_j_plus_1
            valid_change = torch.triu(change, diagonal=2)

            valid_change_flat = valid_change.reshape(batch_size, -1)
            min_change_per_batch = torch.min(valid_change_flat, dim=-1)[0]
            min_change = torch.min(min_change_per_batch)

            if min_change < -eps:
                flatten_argmin_index = torch.argmin(valid_change_flat, dim=-1)
                min_i = torch.div(
                    flatten_argmin_index, num_nodes, rounding_mode="floor"
                )
                min_j = torch.remainder(flatten_argmin_index, num_nodes)

                for b in range(batch_size):
                    if min_change_per_batch[b] < -eps:
                        i_idx = int(min_i[b].item())
                        j_idx = int(min_j[b].item())
                        if i_idx < j_idx:
                            tours[b, i_idx + 1 : j_idx + 1] = torch.flip(
                                tours[b, i_idx + 1 : j_idx + 1], dims=(0,)
                            )
                iterator += 1
            else:
                break

            if iterator >= max_iters:
                break
    
    # Return
    out_dtype = init_tours.dtype if init_tours.dtype in (
        torch.int32,
        torch.int64,
    ) else torch.int32
    return tours.to(out_dtype)
