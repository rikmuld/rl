import random

from itertools import accumulate
from typing import List


def arg_sample(prob_list: List[float]):
    pick = random.random() * sum(prob_list)

    for i, p in enumerate(accumulate(prob_list)):
        if pick < p:
            return i

    return len(prob_list) - 1