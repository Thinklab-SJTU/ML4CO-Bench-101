
import os
import math
import numpy as np
from tensordict.tensordict import TensorDictBase


def prod(sequence):
    """General prod function, that generalised usage across math and np.

    Created for multiple python versions compatibility).

    """
    if hasattr(math, "prod"):
        return math.prod(sequence)
    else:
        return int(np.prod(sequence))
    
    
def seed_generator(seed):
    """A seed generator function.

    Given a seeding integer, generates a deterministic next seed to be used in a
    seeding sequence.

    Args:
        seed (int): initial seed.

    Returns: Next seed of the chain.

    """
    max_seed_val = (
        2**32 - 1
    )  # https://discuss.pytorch.org/t/what-is-the-max-seed-you-can-set-up/145688
    rng = np.random.default_rng(seed)
    seed = int.from_bytes(rng.bytes(8), "big")
    return seed % max_seed_val


def get_binary_env_var(key):
    """Parses and returns the binary environment variable value.

    If not present in environment, it is considered `False`.

    Args:
        key (str): name of the environment variable.
    """
    val = os.environ.get(key, "False")
    if val in ("0", "False", "false"):
        val = False
    elif val in ("1", "True", "true"):
        val = True
    else:
        raise ValueError(
            f"Environment variable {key} should be in 'True', 'False', '0' or '1'. "
            f"Got {val} instead."
        )
    return val


def step_mdp(
    tensordict: TensorDictBase,
    next_tensordict: TensorDictBase = None,
    keep_other: bool = True,
    exclude_reward: bool = False,
    exclude_done: bool = False,
    exclude_action: bool = True,
) -> TensorDictBase:
    """Creates a new tensordict that reflects a step in time of the input tensordict.

    Given a tensordict retrieved after a step, returns the :obj:`"next"` indexed-tensordict.
    THe arguments allow for a precise control over what should be kept and what
    should be copied from the ``"next"`` entry. The default behaviour is:
    move the observation entries, reward and done states to the root, exclude
    the current action and keep all extra keys (non-action, non-done, non-reward).

    Args:
        tensordict (TensorDictBase): tensordict with keys to be renamed
        next_tensordict (TensorDictBase, optional): destination tensordict
        keep_other (bool, optional): if True, all keys that do not start with :obj:`'next_'` will be kept.
            Default is ``True``.
        exclude_reward (bool, optional): if True, the :obj:`"reward"` key will be discarded
            from the resulting tensordict. If ``False``, it will be copied (and replaced)
            from the ``"next"`` entry (if present).
            Default is ``False``.
        exclude_done (bool, optional): if True, the :obj:`"done"` key will be discarded
            from the resulting tensordict. If ``False``, it will be copied (and replaced)
            from the ``"next"`` entry (if present).
            Default is ``False``.
        exclude_action (bool, optional): if True, the :obj:`"action"` key will
            be discarded from the resulting tensordict. If ``False``, it will
            be kept in the root tensordict (since it should not be present in
            the ``"next"`` entry).
            Default is ``True``.

    Returns:
         A new tensordict (or next_tensordict) containing the tensors of the t+1 step.

    Examples:
    This funtion allows for this kind of loop to be used:
        >>> from tensordict import TensorDict
        >>> td = TensorDict({
        ...     "done": torch.zeros((), dtype=torch.bool),
        ...     "reward": torch.zeros(()),
        ...     "extra": torch.zeros(()),
        ...     "next": TensorDict({
        ...         "done": torch.zeros((), dtype=torch.bool),
        ...         "reward": torch.zeros(()),
        ...         "obs": torch.zeros(()),
        ...     }, []),
        ...     "obs": torch.zeros(()),
        ...     "action": torch.zeros(()),
        ... }, [])
        >>> print(step_mdp(td))
        TensorDict(
            fields={
                done: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.bool, is_shared=False),
                extra: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                obs: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                reward: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([]),
            device=None,
            is_shared=False)
        >>> print(step_mdp(td, exclude_done=True))  # "done" is dropped
        TensorDict(
            fields={
                extra: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                obs: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                reward: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([]),
            device=None,
            is_shared=False)
        >>> print(step_mdp(td, exclude_reward=True))  # "reward" is dropped
        TensorDict(
            fields={
                done: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.bool, is_shared=False),
                extra: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                obs: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([]),
            device=None,
            is_shared=False)
        >>> print(step_mdp(td, exclude_action=False))  # "action" persists at the root
        TensorDict(
            fields={
                action: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                done: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.bool, is_shared=False),
                extra: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                obs: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                reward: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([]),
            device=None,
            is_shared=False)
        >>> print(step_mdp(td, keep_other=False))  # "extra" is missing
        TensorDict(
            fields={
                done: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.bool, is_shared=False),
                obs: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                reward: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([]),
            device=None,
            is_shared=False)

    """
    other_keys = []
    prohibited = set()
    if exclude_action:
        prohibited.add("action")
    else:
        other_keys.append("action")
    if exclude_done:
        prohibited.add("done")
    else:
        other_keys.append("done")
    if exclude_reward:
        prohibited.add("reward")
    else:
        other_keys.append("reward")

    prohibited.add("next")
    if keep_other:
        # TODO: make this work with nested keys
        other_keys = [key for key in tensordict.keys() if key not in prohibited]
    select_tensordict = tensordict.select(*other_keys, strict=False)
    excluded = []
    if exclude_reward:
        excluded.append("reward")
    if exclude_done:
        excluded.append("done")
    next_td = tensordict.get("next")
    if len(excluded):
        next_td = next_td.exclude(*excluded)
    select_tensordict = select_tensordict.update(next_td)

    if next_tensordict is not None:
        return next_tensordict.update(select_tensordict)
    else:
        return select_tensordict
