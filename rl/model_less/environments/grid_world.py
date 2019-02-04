from .environment import Environment

from typing import Dict, Any, Tuple
from rl.environments.grid_world import GridWorldBase

class GridWorld(GridWorldBase, Environment):
    def __init__(self, width: int, height: int, win: Tuple[int, int]):
        GridWorldBase.__init__(self, width, height, win)
        Environment.__init__(self,
            [0, 1, 2, 3],
            [(w, h) for w in range(width) for h in range(height)]
        )

    def act(self, action: int):
        reward, state = super(GridWorld, self).act(action)

        if reward is None:
            self.state = self.step(self.state, action)

            if state is self.win:
                self.state = None
                reward = 1
            else:
                reward = 0

        return reward, state