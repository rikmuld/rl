import numpy as np

from .environments.mdp import MDP
from .prediction import evaluate_policy
from .prediction import value_iteration as value_iter

from rl.policy import Policy, Deterministic, Stochastic, is_det_policy
from typing import Dict, Any


def optimal_policy(V: Dict[Any, float], mdp: MDP, gamma: float = 0.9) -> Policy:
    policy = {}

    for s in mdp.states:
        # expected future reward given action (direct reward + the estimated future reward next state)
        EG = lambda a: mdp.E(s, a, lambda sp, r: r + gamma * V[sp])

        # take the action that gives the highest expected future reward
        policy[s] = int(np.argmax([EG(a) for a in mdp.actions]))

    return Deterministic.from_dict(policy)


def policy_iteration(mdp: MDP, policy: Policy = None,
                     gamma: float = 0.9, start_uniform: bool = True,
                     delta: float = 0.05, printit: bool = True) -> Policy:
    if policy is None:
        if start_uniform:
            policy = Stochastic.uniform_policy(len(mdp.actions))
        else:
            policy = Deterministic.constant_policy(mdp.actions[0])

    all_same = False

    while not all_same:
        all_same = is_det_policy(policy)

        V = evaluate_policy(policy, mdp, gamma, delta)
        policy_new = optimal_policy(V, mdp, gamma)

        if all_same:
            for s in mdp.states:
                if policy(s) is not policy_new(s):
                    all_same = False
                    break

        policy = policy_new

        if printit:
            mdp.visualize(V=V, policy=policy)

    return policy


def value_iteration(mdp: MDP, gamma: float = 0.9, delta: float = 0.05, printit: bool = True) -> Policy:
    V = value_iter(mdp, gamma, printit, delta)
    policy = optimal_policy(V, mdp, gamma)

    if printit:
        mdp.visualize(policy=policy)

    return policy