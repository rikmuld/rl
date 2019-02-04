import numpy as np

from typing import List, Any


class Environment:
    def __init__(self, actions: List[int], states: List[Any]):
        self.actions = actions
        self.states = states

        self.state = None

    def act(self, action: int):
        assert (action in self.actions)

        if self.state is None:
            return 0, None
        else:
            return None, None

    def reset(self, state=None):
        assert (state is None or state in self.states)

        if state is None:
            self.state = np.random.choice(self.states)
        else:
            self.state = state

        return self.state

    def visualize(self, V=None, policy=None):
        """Optional; add visualization of environment here"""
        raise NotImplementedError