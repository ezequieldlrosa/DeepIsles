from operator import truth
from typing import Optional, Sequence, Hashable, Tuple, Union
from numbers import Number, Real
import math

from sympy import Symbol, solve

from .base import Nodes, TensorNetwork
from .utils import prod


class SingleTensor(TensorNetwork):
    def __init__(
        self,
        size: Sequence[int],
        legs: Optional[Sequence[Hashable]] = None,
        name: Optional[Hashable] = "X",
        **kwargs,
    ) -> None:
        self.size = size
        if legs is None:
            legs = [f"i_{j}" for j in range(len(self.size))]

        nodes: Nodes = {name: {"shape": self.size, "legs": legs}}
        super().__init__(nodes=nodes, output_edges=legs, **kwargs)


class CanonicalPolyadic(TensorNetwork):
    def __init__(
        self,
        size: Sequence[int],
        rank: Optional[int] = None,
        compression: Optional[Number] = None,
        batch: Optional[bool] = True,
        **kwargs,
    ) -> None:

        assert (rank, compression) != (
            None,
            None,
        ), "'rank' or 'compression' must be specified."

        self.size = size

        if rank is None:
            self.rank = self._get_rank(size, compression)
        else:
            self.rank = rank

        if batch:
            batch_dim = (1,)
            batch_leg = ["b"]
        else:
            batch_dim = ()
            batch_leg = []

        nodes: Nodes = {}
        output_edges = [*batch_leg]
        for j, s in enumerate(size):
            nodes[f"U_{j}"] = {
                "shape": (*batch_dim, s, self.rank),
                "legs": [*batch_leg, f"i_{j}", "r"],
                "tags": {"factor_matrix"},
            }
            output_edges.append(f"i_{j}")

        super().__init__(nodes=nodes, output_edges=output_edges, **kwargs)

    def _get_rank(self, size, compression) -> int:
        # degrees of freedom of the input tensor
        df_input = prod(size)
        # degrees of freedom of the low-rank tensor
        df_lowrank = sum(size)
        rank = max(int(math.ceil(df_input / (compression * df_lowrank))), 1)
        return rank


class Tucker(TensorNetwork):
    def __init__(
        self,
        size,
        rank: Optional[Sequence[int]] = None,
        compression: Optional[Number] = None,
        batch: Optional[bool] = True,
        **kwargs,
    ) -> None:

        assert (rank, compression) != (
            None,
            None,
        ), "'rank' or 'compression' must be specified."

        self.size = size

        if rank is None:
            self.rank = self._get_rank(size, compression)
        else:
            self.rank = rank

        self.rank = self._check_and_update_rank(self.size, self.rank)

        if batch:
            batch_dim = (1,)
            batch_leg = ["b"]
        else:
            batch_dim = ()
            batch_leg = []

        nodes: Nodes = {}
        output_edges = [*batch_leg]
        for j, s in enumerate(self.size):
            nodes[f"U_{j}"] = {
                "shape": (*batch_dim, s, self.rank[j]),
                "legs": [*batch_leg, f"i_{j}", f"r_{j}"],
                "tags": {"factor_matrix"},
            }

            output_edges.append(f"i_{j}")

        nodes["G"] = {
            "shape": (*batch_dim, *self.rank),
            "legs": [*batch_leg, *(f"r_{j}" for j in range(len(self.size)))],
            "tags": {"core_tensor"},
        }

        super().__init__(nodes=nodes, output_edges=output_edges, **kwargs)

    def _get_rank(self, size, compression) -> Sequence[int]:
        ndims = len(size)
        # degrees of freedom of the input tensor
        df_input = prod(size)
        # alpha is the dim ratio
        alpha = Symbol("alpha", real=True)
        # degrees of freedom of the low-rank tensor
        df_lowrank = sum(s ** 2 for s in size) / alpha + prod(size) / (
            alpha ** ndims
        )

        alpha = solve(df_input - compression * df_lowrank, alpha)
        alpha = min(
            a.evalf()
            for a in alpha
            if isinstance(a.evalf(), Real) and a.evalf() > 0
        )
        rank = tuple(int(max(math.ceil(s / alpha), 2)) for s in size)
        return rank

    def _check_and_update_rank(self, size, rank):
        """Check theoretical bound for multilinear rank."""

        rank = list(rank)
        satisfied = False
        while not satisfied:
            satisfied = True
            for j, s in enumerate(size):
                rest_rank = list(rank)
                r = rest_rank.pop(j)
                upper_bound = min(s, prod(rest_rank))
                if r > upper_bound:
                    rank[j] = upper_bound
                    satisfied = False

        return rank


class TensorTrain(TensorNetwork):
    def __init__(
        self,
        size: Sequence[int],
        rank: Optional[Sequence[int]] = None,
        compression: Optional[Number] = None,
        batch: Optional[bool] = True,
        **kwargs,
    ) -> None:

        assert (rank, compression) != (
            None,
            None,
        ), "'rank' or 'compression' must be specified."

        self.size = size

        if rank is None:
            self.rank = self._get_rank(size, compression)
        else:
            self.rank = rank

        if batch:
            batch_dim = (1,)
            batch_leg = ["b"]
        else:
            batch_dim = ()
            batch_leg = []

        nodes: Nodes = {}
        output_edges = [*batch_leg]
        for j, s in enumerate(self.size):
            if j == 0:
                shape = (*batch_dim, s, self.rank[j])
                legs = [*batch_leg, f"i_{j}", f"r_{j}"]
                tags = {"factor_matrix"}
            elif j == len(size) - 1:
                shape = (*batch_dim, s, self.rank[j - 1])
                legs = [*batch_leg, f"i_{j}", f"r_{j-1}"]
                tags = {"factor_matrix"}
            else:
                shape = (*batch_dim, s, self.rank[j - 1], self.rank[j])
                legs = [*batch_leg, f"i_{j}", f"r_{j-1}", f"r_{j}"]
                tags = {"factor_tensor"}

            nodes[f"U_{j}"] = {
                "shape": shape,
                "legs": legs,
                "tags": tags,
            }

            output_edges.append(f"i_{j}")

        super().__init__(nodes=nodes, output_edges=output_edges, **kwargs)

    def _get_rank(self, size, compression) -> Sequence[int]:
        # degrees of freedom of the input tensor
        df_input = prod(size)
        # rank (or bond dimension) is the same for all
        rank = Symbol("rank", real=True)
        # degrees of freedom of the low-rank tensor
        df_lowrank = (size[0] + size[-1]) * rank + sum(size[1:-1]) * rank ** 2

        rank = solve(df_input - compression * df_lowrank, rank)
        rank = min(
            a.evalf()
            for a in rank
            if isinstance(a.evalf(), Real) and a.evalf() > 0
        )
        rank = tuple(max(int(math.ceil(rank)), 2) for s in size[:-1])
        return rank
