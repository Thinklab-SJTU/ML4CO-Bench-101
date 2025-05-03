import torch
from torch import Tensor


def remove_origin_and_reorder_tensor(tensor: Tensor, is_selected: Tensor) -> Tensor:
    """
        From batch of vectors remove origin and put selected element to the beginning
    """
    bs = tensor.shape[0]
    # remove original idx of selected node
    new_beginning = tensor[is_selected].unsqueeze(dim=1)
    new_remaining = tensor[~is_selected].reshape((bs, -1))[:, 1:]
    new_tensor = torch.cat([new_beginning, new_remaining], dim=1)
    return new_tensor


def remove_origin_and_reorder_matrix(matrices: Tensor, is_selected: Tensor) -> Tensor:
    """
        From batch of vectors remove origin and put selected element to the beginning
    """
    bs, subpb_size, _, num_features = matrices.shape
    # select row (=column) of adj matrix for just-selected node
    selected_row = matrices[is_selected]
    selected_column = matrices.transpose(1, 2)[is_selected]
    # remove distance to the selected node (=0)
    selected_row = selected_row[~is_selected].reshape((bs, -1, num_features))[:, 1:]
    selected_column = selected_column[~is_selected].reshape((bs, -1, num_features))[:, 1:]

    # remove rows and columns of selected nodes
    remaining_matrices = matrices[~is_selected].reshape(bs, -1, subpb_size, num_features)[:, 1:, :]
    remaining_matrices = remaining_matrices.transpose(1, 2)[~is_selected].reshape(bs, subpb_size - 1, -1, num_features)[:, 1:, :]
    remaining_matrices = remaining_matrices.transpose(1, 2)

    # add new row on the top and remove second (must be done like this, because on dimensions of the matrix)
    remaining_matrices = torch.cat([selected_row.unsqueeze(dim=1), remaining_matrices], dim=1)

    # and add it to the beginning-
    new_matrices = torch.cat([torch.zeros([bs, 1, num_features], device=selected_row.device), selected_column],
                             dim=1)
    new_matrices = torch.cat([new_matrices.unsqueeze(dim=2), remaining_matrices], dim=2)

    return new_matrices
