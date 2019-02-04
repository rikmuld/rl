
from typing import Dict, Any, Tuple
from rl.policy import Policy, is_det_policy
from rl.utils.string import pad

class GridWorldBase():
    def __init__(self, width: int, height: int, win: Tuple[int, int]):
        self.win = win
        self.width = width
        self.height = height

    def step(self, s: Tuple[int, int], a: int):
        if s == self.win:
            return self.win

        if a == 0:
            return (max(0, s[0] - 1), s[1])
        if a == 1:
            return (min(self.width - 1, s[0] + 1), s[1])
        if a == 2:
            return (s[0], max(0, s[1] - 1))
        if a == 3:
            return (s[0], min(self.height - 1, s[1] + 1))

    def visualize(self, V: Dict[Any, float] = None, policy: Policy = None):
        if V is not None:
            self.show_fn(lambda w, h: pad(round(V[(w, h)], 2), 4, "0"))

        if policy is not None:
            if is_det_policy(policy):
                self.show_fn(lambda w, h: policy((w, h)))
            else:
                print("Only deterministic policies can be visualized!")

        if policy is None and V is None:
            self.show_fn(lambda w, h: "x" if (w, h) == self.win else " ")

    def show_fn(self, f):
        for h in range(self.height):
            print("| ", end="")
            for w in range(self.width):
                a = f(w, h)

                print(str(a) + " | ", end="")
            print()
        print()