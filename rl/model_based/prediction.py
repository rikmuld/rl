from .environments.mdp import MDP

from rl.policy import Policy, is_det_policy
from typing import Dict, Any


def evaluate_policy(policy: Policy, mdp: MDP, gamma=0.9, delta=0.05, printit: bool = False) -> Dict[Any, float]:
    """Approximates the states values of the policy"""

    def update_policy(s, V, EG):
        if is_det_policy(policy):
            V[s] = EG(policy(s))
        else:
            V[s] = sum([policy(a, s) * EG(a) for a in mdp.actions])

    return iterative_value_update(update_policy, mdp, gamma, delta, printit)


def value_iteration(mdp: MDP, gamma: float = 0.9, printit: bool = False, delta: float = 0.05) -> Policy:
    """Approximates the optimal states values"""

    def update_policy(s, V, EG):
        V[s] = max([EG(a) for a in mdp.actions])

    return iterative_value_update(update_policy, mdp, gamma, delta, printit)


def iterative_value_update(update_rule, mdp: MDP, gamma=0.9, delta=0.05, printit: bool = False) -> Dict[Any, float]:
    V = {s: 0.0 for s in mdp.states}
    do_break = False

    while not do_break:
        do_break = True

        for s in mdp.states:
            v = V[s]
            EG = lambda a: mdp.E(s, a, lambda sp, r: r + gamma * V[sp])  # expected future reward given action

            update_rule(s, V, EG)

            if printit:
                mdp.visualize(V = V)

            if abs(V[s] - v) > delta:
                do_break = False

    return V
