from .mdp import MDP

from typing import Dict, Any, Tuple
from rl.environments.grid_world import GridWorldBase

class GridWorld(GridWorldBase, MDP):
    def __init__(self, width: int, height: int, win: Tuple[int, int]):
        GridWorldBase.__init__(self, width, height, win)
        MDP.__init__(self,
            [0, 1, 2, 3],
            [(w, h) for w in range(width) for h in range(height)],
            [1, 0]
        )

    def p(self, sp, r, s, a):
        super(GridWorld, self).p(sp, r, s, a)

        sp_should = self.step(s, a)
        r_should = int(sp_should == self.win and s != self.win)

        return 1 if sp_should == sp and r_should == r else 0