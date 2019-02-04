
from typing import Union

from rl.model_based.environments.mdp import MDP
from rl.model_less.environments.environment import Environment
from rl.policy import Policy, Deterministic, Stochastic, is_det_policy


def policy_iteration(env: Union[Environment, MDP], eval_policy_fn, optimal_policy_fn, policy: Policy = None,
                     start_uniform: bool = True, printit: bool = True) -> Policy:
    if policy is None:
        if start_uniform:
            policy = Stochastic.uniform_policy(len(env.actions))
        else:
            policy = Deterministic.constant_policy(env.actions[0])

    all_same = False

    while not all_same:
        all_same = is_det_policy(policy)

        V = eval_policy_fn(policy, env)
        policy_new = optimal_policy_fn(V, env)

        if all_same:
            for s in env.states:
                if policy(s) is not policy_new(s):
                    all_same = False
                    break

        policy = policy_new

        if printit:
            env.visualize(V=V, policy=policy)

    return policy