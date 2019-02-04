from rl.policy import Policy, sample_action
from typing import Dict, Any
from .environments.environment import Environment


def evaluate_policy(policy: Policy, env: Environment, gamma=0.9, alpha: float = 0.1,
                    episodes: int = 10, printit: bool = False, state_action: bool = False) -> Dict[Any, Any]:

    V = {s: 0.0 for s in env.states}

    for _ in range(episodes):
        s = env.reset()

        while env.state is not None:
            a = sample_action(policy, env.state, env.actions)
            r, sp = env.act(a)

            V[s] = V[s] + alpha * (r + gamma * V[sp] - V[s])

            s = sp

        if printit:
            env.visualize(V=V)

    return V
