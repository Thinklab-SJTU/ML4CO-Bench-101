import torch
from torch import Tensor
from typing import Union
from einops import rearrange
from tensordict import TensorDict


def _batchify_single(
    x: Union[Tensor, TensorDict], repeats: int
) -> Union[Tensor, TensorDict]:
    """Same as repeat on dim=0 for Tensordicts as well"""
    s = x.shape
    return x.expand(repeats, *s).contiguous().view(s[0] * repeats, *s[1:])


def batchify(
    x: Union[Tensor, TensorDict], shape: Union[tuple, int]
) -> Union[Tensor, TensorDict]:
    shape = [shape] if isinstance(shape, int) else shape
    for s in reversed(shape):
        x = _batchify_single(x, s) if s > 0 else x
    return x


def _unbatchify_single(
    x: Union[Tensor, TensorDict], repeats: int
) -> Union[Tensor, TensorDict]:
    s = x.shape
    return x.view(repeats, s[0] // repeats, *s[1:]).permute(1, 0, *range(2, len(s) + 1))


def unbatchify(
    x: Union[Tensor, TensorDict], shape: Union[tuple, int]
) -> Union[Tensor, TensorDict]:
    shape = [shape] if isinstance(shape, int) else shape
    for s in reversed(
        shape
    ):  # we need to reverse the shape to unbatchify in the right order
        x = _unbatchify_single(x, s) if s > 0 else x
    return x


def gather_by_index(src, idx, dim=1, squeeze=True):
    expanded_shape = list(src.shape)
    expanded_shape[dim] = -1
    idx = idx.view(idx.shape + (1,) * (src.dim() - idx.dim())).expand(expanded_shape)
    squeeze = idx.size(dim) == 1 and squeeze
    return src.gather(dim, idx).squeeze(dim) if squeeze else src.gather(dim, idx)


def unbatchify_and_gather(x: Tensor, idx: Tensor, n: int):
    x = unbatchify(x, n)
    return gather_by_index(x, idx, dim=idx.dim())


def get_distance(x: Tensor, y: Tensor):
    return (x - y).norm(p=2, dim=-1)


def get_tour_length(ordered_locs):
    ordered_locs_next = torch.roll(ordered_locs, -1, dims=-2)
    return get_distance(ordered_locs_next, ordered_locs).sum(-1)


def get_distance_matrix(locs: Tensor):
    distance = (locs[..., :, None, :] - locs[..., None, :, :]).norm(p=2, dim=-1)
    return distance


def calculate_entropy(logprobs: Tensor):
    logprobs = torch.nan_to_num(logprobs, nan=0.0)
    entropy = -(logprobs.exp() * logprobs).sum(dim=-1)  # [batch, decoder steps]
    entropy = entropy.sum(dim=1)  # [batch] -- sum over decoding steps
    assert entropy.isfinite().all(), "Entropy is not finite"
    return entropy


def select_start_nodes(td, env, num_starts):
    num_loc = env.generator.num_loc if hasattr(env.generator, "num_loc") else 0xFFFFFFFF
    if env.name in ["tsp", "atsp", "flp", "mcp"]:
        selected = (
            torch.arange(num_starts, device=td.device).repeat_interleave(td.shape[0])
            % num_loc
        )
    elif env.name in ["jssp", "fjsp"]:
        raise NotImplementedError("Multistart not yet supported for FJSP/JSSP")
    else:
        # Environments with depot: we do not select the depot as a start node
        selected = (
            torch.arange(num_starts, device=td.device).repeat_interleave(td.shape[0])
            % num_loc
            + 1
        )
        if env.name == "op":
            if (td["action_mask"][..., 1:].float().sum(-1) < num_starts).any():
                # for the orienteering problem, we may have some nodes that are not available
                # so we need to resample from the distribution of available nodes
                selected = (
                    torch.multinomial(
                        td["action_mask"][..., 1:].float(), num_starts, replacement=True
                    )
                    + 1
                )  # re-add depot index
                selected = rearrange(selected, "b n -> (n b)")
    return selected


def get_best_actions(actions, max_idxs):
    actions = unbatchify(actions, max_idxs.shape[0])
    return actions.gather(0, max_idxs[..., None, None])


def get_num_starts(td: TensorDict) -> Tensor:
    """Returns the number of possible start nodes for the environment based on the action mask"""
    return td["action_mask"].shape[-1]


def select_start_nodes(td: TensorDict, nodes_num: int = None):
    """Node selection strategy as proposed in POMO (Kwon et al. 2020)
    and extended in SymNCO (Kim et al. 2022).
    Selects different start nodes for each batch element

    Args:
        td: TensorDict containing the data. We may need to access the available actions to select the start nodes
        nodes_num: Number of nodes to select
    """
    nodes_num = get_num_starts(td) if nodes_num is None else nodes_num

    # Environments with depot: don't select the depot as start node
    selected = torch.arange(nodes_num, device=td.device).repeat_interleave(td.shape[0])
    return selected