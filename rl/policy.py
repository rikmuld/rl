from typing import NewType, Dict, Callable, Union, Any, List
from inspect import signature


Policy = NewType("Policy", Union[Callable[[int, Any], float], Callable[[Any], int]])


def is_det_policy(policy: Policy):
    return len(signature(policy).parameters) is 1


def is_stoch_policy(policy: Policy):
    return not is_det_policy(policy)


# pinch eyes and say namespace
class Deterministic:
    @staticmethod
    def from_dict(state_to_action: Dict[Any, int]):
        return Policy(lambda s: state_to_action[s])

    @staticmethod
    def constant_policy(a: int):
        return Policy(lambda s: a)


# pinch eyes and say namespace
class Stochastic:
    @staticmethod
    def constant_policy(c: float):
        return Policy(lambda a, s: c)

    @staticmethod
    def uniform_policy(actions: int):
        return Stochastic.constant_policy(1 / actions)