import math

import torch
from torch import nn

from ..utils.helpers import wrap_class, null_context
from .operations import t, relative_error
from ..tensor_network import SingleTensor


class MF(nn.Module):
    """Base module for matrix factorization.

    X ≈ U t(V),
    U, V ∈ S
    """

    def __init__(
        self,
        size,
        rank=None,
        compression=10,
        num_iters=5,
        num_grad_steps=None,
        init=None,
        solver=None,
        verbose=False,
        **kwargs,
    ):
        super().__init__()
        self.size = M, N = size
        self.num_iters = num_iters
        self.num_grad_steps = (
            num_iters if num_grad_steps is None else num_grad_steps
        )

        assert (rank, compression) != (
            None,
            None,
        ), "'rank' or 'compression' must be specified."

        # degrees of freedom of the input matrix
        df_input = M * N
        # degrees of freedom of the low-rank matrix
        df_lowrank = M + N

        if rank is None:
            self.rank = rank = max(
                math.ceil(df_input / (compression * df_lowrank)), 1
            )
        else:
            self.rank = rank

        # update compression ratio
        self.compression = df_input / (self.rank * df_lowrank)

        # set factors initializer
        init = wrap_class(init)
        self.init = init(size=size, rank=rank)

        # set solver
        solver = wrap_class(solver)
        self.solver = solver(size=size, rank=rank)

        # FLOPs
        self.flops = {
            "init": getattr(self.init, "flops", None),
            "decompose": num_iters * self.solver.flops
            if hasattr(self.solver, "flops")
            else None,
            "reconstruct": 2 * M * rank * N,
        }

        self.verbose = verbose

    def context(self, it):
        # get context at each iteration it
        if it < self.num_iters - self.num_grad_steps + 1:
            context = torch.no_grad()
        else:
            context = null_context()

        return context

    def decompose(self, x, *args, **kwargs):
        # x: B × M × N

        # initialize
        with self.context(0):
            u, v = self.init(x)

        # iterate
        for it in range(1, self.num_iters + 1):
            with self.context(it):
                if self.verbose:
                    loss = self.loss(x, u, v)
                    print(f"iter {it}, loss = {loss}")

                u, v = self.solver(x, [u, v], *args, **kwargs)

        return u, v

    def reconstruct(self, u, v):
        return u @ t(v)

    def loss(self, x, u, v, w=None):
        return relative_error(x, self.reconstruct(u, v), w)

    def forward(self, x):
        u, v = self.decompose(x)
        return self.reconstruct(u, v)


class TF(nn.Module):
    """Base module for tensor factorization.

    Decompose a tensor X into an arbitrary tensor network, i.e.,
    X ≈ TensorNework(U1, ..., UM)
    """

    def __init__(
        self,
        tensor_network,
        num_iters=5,
        num_grad_steps=None,
        trainable_dims=(),
        init=None,
        solver=None,
        contract_params=None,
        verbose=False,
        **kwargs,
    ):
        super().__init__()
        self.tensor_network = tensor_network
        self.num_iters = num_iters
        self.num_grad_steps = (
            num_iters if num_grad_steps is None else num_grad_steps
        )

        # set factors initializer
        init = wrap_class(init)
        self.init = init(tensor_network=tensor_network)

        # set solver
        solver = wrap_class(solver)
        self.solver = solver(tensor_network=tensor_network)

        # set contractor
        if contract_params is None:
            self.contract_params = {}
        else:
            self.contract_params = contract_params

        self.contract_params.setdefault("optimize", "optimal")

        # make contractor for reconstruction (without trainable weights)
        (
            self.reconstruct_expr,
            self.reconstruct_info,
        ) = self.tensor_network.contract_expression(**self.contract_params)

        # make contractor for forward path, which has trainable weights
        self.trainable_dims = trainable_dims
        if any(True for _ in self.trainable_dims):
            self.tn = self.tensor_network.copy()
            output_edges = list(self.tensor_network.output_edges)
            for j, e in enumerate(self.tensor_network.output_edges[1:]):
                if j in self.trainable_dims:
                    d = self.tensor_network.edges[e]["dimension"]
                    matrix = SingleTensor((d, d), (e, f"{e}_"), f"W{j}")
                    self.tn.join(matrix, inplace=True)
                    setattr(self, f"W{j}", nn.Parameter(torch.randn(d, d)))
                    output_edges[j + 1] = f"{e}_"

            self.tn.output_edges = output_edges
            self.forward_expr, self.forward_info = self.tn.contract_expression(
                **self.contract_params
            )
        else:
            self.forward_expr = self.reconstruct_expr
            self.forward_info = self.reconstruct_info

        # FLOPs
        self.flops = {}
        # initialization flops
        self.flops["init"] = getattr(self.init, "flops", None)
        # decomposition flops
        self.flops["decompose"] = (
            num_iters * self.solver.flops
            if hasattr(self.solver, "flops")
            else None
        )
        # reconstruction flops
        self.flops["reconstruct"] = self.forward_info.opt_cost

        self.compression = self.tensor_network.compression
        self.verbose = verbose

    def context(self, it):
        # get context at each iteration it
        if it < self.num_iters - self.num_grad_steps + 1:
            context = torch.no_grad()
        else:
            context = null_context()

        return context

    def decompose(self, x):
        # x: B × N1 × N2 × ... × Np

        # initialize
        with self.context(0):
            factors = self.init(x)

        # iterate
        for it in range(1, self.num_iters + 1):
            with self.context(it):
                if self.verbose:
                    loss = self.loss(x, factors)
                    print(f"iter {it}, loss = {loss}")

                factors = self.solver(x, factors)

        return factors

    def reconstruct(self, factors):
        return self.reconstruct_expr(factors)

    def loss(self, x, factors):
        return relative_error(x, self.reconstruct(factors))

    def forward(self, x):
        factors = self.decompose(x)
        tensors = {**factors, **dict(self.named_parameters())}
        return self.forward_expr(tensors)

