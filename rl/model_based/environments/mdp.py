from typing import List, Any


class MDP:
    def __init__(self, actions: List[int], states: List[Any], rewards: List[float]):
        self.actions = actions
        self.states = states
        self.rewards = rewards

    def p(self, sp, r, s, a):
        """Definition of the MDP, must be overridden, call for safety checks"""
        assert (sp in self.states)
        assert (s in self.states)
        assert (r in self.rewards)
        assert (a in self.actions)

        return None

    def state_transition_prob(self, sp, s, a):
        return sum([self.p(sp, r, s, a) for r in self.rewards])

    def expected_reward(self, s, a):
        return sum([r * sum([self.p(sp, r, s, a) for sp in self.states]) for r in self.rewards])

    def E(self, s, a, f):
        """Calculates an expectation under the conditional probability p(sp, r|s, a); f takes sp and r"""
        return sum([self.p(sp, r, s, a) * f(sp, r) for sp in self.states for r in self.rewards])

    def visualize(self, V=None, policy=None):
        """Optional; add visualization of environment here"""
        raise NotImplementedError
