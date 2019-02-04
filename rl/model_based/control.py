import numpy as np

from .environments.mdp import MDP
from .prediction import evaluate_policy

from rl.policy import Policy, Deterministic, Stochastic, is_det_policy
from typing import Dict, Any

from rl.control import policy_iteration as policy_iter


def optimal_policy(V: Dict[Any, float], mdp: MDP, gamma: float = 0.9) -> Policy:
    policy = {}

    for s in mdp.states:
        EG = lambda a: mdp.E(s, a, lambda sp, r: r + gamma * V[sp])
        policy[s] = int(np.argmax([EG(a) for a in mdp.actions]))

    return Deterministic.from_dict(policy)


def policy_iteration(mdp: MDP, policy: Policy = None,
                     gamma: float = 0.9, start_uniform: bool = True,
                     delta: float = 0.05, printit: bool = True) -> Policy:

    return policy_iter(mdp,
                       lambda p, mdp: evaluate_policy(policy, mdp, gamma, delta),
                       lambda V, mdp: optimal_policy(V, mdp, gamma),
                       policy,
                       start_uniform,
                       printit)