from typing import Sequence
from numbers import Number
from functools import reduce
from operator import mul


def prod(x: Sequence[Number]):
    return reduce(mul, x, 1)

