import numpy as np

from typing import List


class Environment:
    def __init__(self, actions: List[int], states: List[int]):
        self.actions = actions
        self.states = states + [None]

        self.state = None

    def act(self, action: int):
        assert (action in self.actions)

        if self.state is None:
            return 0, None

    def reset(self, state=None):
        assert (state in self.states)

        if state is None:
            self.state = np.random.choice(self.states)
        else:
            self.state = state

        return self.state