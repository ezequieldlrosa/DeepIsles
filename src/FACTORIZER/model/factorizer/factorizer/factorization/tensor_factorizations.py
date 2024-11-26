import math
from itertools import permutations

import torch
from torch import nn
import einops

from ..utils.helpers import as_tuple, null_context, prod
from .base import TF
from ..tensor_network import (
    SingleTensor,
    CanonicalPolyadic,
    Tucker,
    TensorTrain,
)


################################
# Initializers
################################


class RandomTensorInit(nn.Module):
    def __init__(self, tensor_network, method="uniform", **kwargs):
        super().__init__()
        self.tensor_network = tensor_network
        self.method = getattr(nn.init, f"{method}_")

        # register factors
        for k, v in tensor_network.nodes.items():
            self.register_buffer(k, torch.empty(v["shape"]))

        # init factors
        for k in tensor_network.nodes:
            self.method(getattr(self, k), **kwargs)

        self.flops = 0

    def forward(self, x):
        # expand initial factors along the batch dim.
        B = x.shape[0]
        factors = {}
        for k in self.tensor_network.nodes:
            u = einops.repeat(getattr(self, k), "1 ... -> b ...", b=B)
            factors[k] = u

        return factors


class MLSVDInit(nn.Module):
    def __init__(self, tensor_network, nonnegative=False):
        super().__init__()
        self.tensor_network = tensor_network
        self.nonnegative = nonnegative

        # init MLSVD
        self.mlsvd = MLSVD(tensor_network.size, rank=tensor_network.rank)

        self.flops = self.mlsvd.flops["decompose"]

    def forward(self, x):
        factors = self.mlsvd.decompose(x)
        for k in self.tensor_network.nodes:
            if self.nonnegative:
                factors[k] = torch.relu(factors[k])

        return factors


################################
# Solvers
################################


class GeneralizedMultiplicativeUpdate(nn.Module):
    def __init__(
        self, tensor_network, nodes=None, eps=1e-8, contract_params=None
    ):
        super().__init__()
        self.tensor_network = self.tn = tensor_network
        if nodes is None:
            self.nodes = list(self.tn.nodes)
        else:
            self.nodes = nodes

        self.nodes = as_tuple(self.nodes)
        self.eps = eps
        if contract_params is None:
            self.contract_params = {}
        else:
            self.contract_params = contract_params

        self.contract_params.setdefault("optimize", "optimal")

        # make contraction experssions
        self.numerator_tn = dict.fromkeys(self.nodes)
        self.numerator_expr = dict.fromkeys(self.nodes)
        self.denominator_tn = dict.fromkeys(self.nodes)
        self.denominator_expr = dict.fromkeys(self.nodes)
        self.flops = 0
        for node in self.nodes:
            self.numerator_tn[node] = self.make_numerator_tn(self.tn, node)
            self.numerator_expr[node] = self.numerator_tn[
                node
            ].contract_expression(**self.contract_params)
            self.flops += self.numerator_expr[node][1].opt_cost
            self.denominator_tn[node] = self.make_denominator_tn(self.tn, node)
            self.denominator_expr[node] = self.denominator_tn[
                node
            ].contract_expression(**self.contract_params)
            self.flops += self.denominator_expr[node][1].opt_cost

    def make_numerator_tn(self, tn, node):
        numerator_tn = SingleTensor(tn.output_shape, tn.output_edges)
        numerator_tn.join(
            tn,
            linking_edges=(numerator_tn.output_edges, tn.output_edges),
            inplace=True,
        )
        numerator_tn.output_edges = numerator_tn.nodes[node]["legs"]
        return numerator_tn

    def make_denominator_tn(self, tn, node):
        denominator_tn = tn.copy()
        denominator_tn.rename_nodes({n: f"{n}_*" for n in tn.nodes})
        denominator_tn.rename_edges(
            {e: f"{e}_*" for e in tn.edges if e not in tn.output_edges}
        )
        denominator_tn.join(
            self.tn,
            linking_edges=(denominator_tn.output_edges, tn.output_edges),
            inplace=True,
        )
        removed_node = denominator_tn.popnode(node)
        denominator_tn.output_edges = removed_node.nodes[node]["legs"]
        return denominator_tn

    def forward(self, x, tensors_dict):
        for node in self.nodes:
            numerator_expr, _ = self.numerator_expr[node]
            numerator = numerator_expr({"X": x, **tensors_dict}) + self.eps

            denominator_tensors_dict = {}
            for k in self.denominator_tn[node].nodes:
                if k in tensors_dict:
                    denominator_tensors_dict[k] = tensors_dict[k]
                else:
                    # k[:-2] removes '_*' from the end of the node name
                    denominator_tensors_dict[k] = tensors_dict[k[:-2]]

            denominator_expr, _ = self.denominator_expr[node]
            denominator = denominator_expr(denominator_tensors_dict) + self.eps

            tensors_dict[node] = numerator / denominator

        return tensors_dict


class CPMultiplicativeUpdate(GeneralizedMultiplicativeUpdate):
    def __init__(
        self, size, rank=None, factor=None, **kwargs,
    ):
        canonical_polyadic = CanonicalPolyadic(size, rank)
        node_names = list(canonical_polyadic.nodes.keys())
        nodes = [node_names[j] for j in as_tuple(factor)]
        super().__init__(canonical_polyadic, nodes, **kwargs)

    def forward(self, x, tensors):
        tensors_dict = dict([*zip(self.tn.nodes, tensors)])
        return tuple(super().forward(x, tensors_dict).values())


###################################
# Tensor factorizers
###################################


class MLSVD(TF):
    """Multilinear Singular Value Decomposition.

        X ≈ (G; U1, ..., UM)
    """

    def __init__(
        self,
        size,
        rank=None,
        compression=10,
        no_grad=False,
        contract_params=None,
        verbose=False,
        **kwargs,
    ):
        self.size = size
        self.tucker = Tucker(size, rank, compression)
        self.ndims = len(size)
        self.rank = self.tucker.rank
        self.no_grad = no_grad

        def init(tensor_network):
            def wrapper(x):
                return {k: None for k in self.tucker.nodes}

            wrapper.flops = 0
            return wrapper

        # compute the cheapest order of factors
        flops_dict = {}
        for order in permutations(range(self.ndims)):
            flops = 0
            shape = list(size)
            for j in order:
                flops += self.sinle_update_flops(shape, j, self.rank[j])
                shape[j] = self.rank[j]

            flops_dict[order] = flops

        self.order = min(flops_dict, key=flops_dict.get)
        self.flops = flops_dict[self.order]

        def solver(tensor_network):
            def update(x, factors):
                core = x
                for j in self.order:
                    u, core = self.sinle_update(core, j, self.rank[j])
                    factors[f"U_{j}"] = u

                factors["G"] = core
                return factors

            update.flops = self.flops
            return update

        super().__init__(
            self.tucker,
            num_iters=0,
            num_grad_steps=0,
            init=init,
            solver=solver,
            contract_params=contract_params,
            verbose=verbose,
            **kwargs,
        )

    def sinle_update(self, core, dim, rank):
        ndims = core.ndim - 1
        shape = core.shape[1:]

        inds1 = "b "
        inds1 += " ".join([f"i_{j}" for j in range(ndims)])
        inds2 = f"b i_{dim} "
        inds2 += f'({" ".join([f"i_{j}" for j in range(ndims) if j!=dim])})'
        core_mat = einops.rearrange(core, f"{inds1} -> {inds2}")

        torch.manual_seed(42)
        u, s, v = torch.svd_lowrank(core_mat, rank)
        core_new = torch.einsum("br, bjr -> brj", s, v)

        dim_dict = {f"i_{j}": shape[j] for j in range(ndims)}
        dim_dict[f"i_{dim}"] = rank
        core_new = einops.rearrange(
            core_new, f"{inds2} -> {inds1}", **dim_dict
        )
        return u, core_new

    def sinle_update_flops(self, size, dim, rank):
        size_ = list(size)
        m = size_.pop(dim)
        n = prod(size_)

        flops = 6 * m * n * rank + (m + n) * (rank ** 2)
        flops += n * (rank ** 2)
        return 2 * math.ceil(flops)

    def context(self):
        # get context
        if self.no_grad:
            context = torch.no_grad()
        else:
            context = null_context()

        return context

    def decompose(self, x):
        # x: B × N1 × N2 × ... × Np

        with self.context():
            # initialize
            factors = self.init(x)

            # iterate
            factors = self.solver(x, factors)

            if self.verbose:
                loss = self.loss(x, factors)
                print(f"loss = {loss}")

        return factors


class NCPD(TF):
    """Non-negative Canonical Polyadic Decomposition (NTF).

        X ≈ (U1, ..., UM),
        U1, ..., UM ≥ 0
    """

    def __init__(
        self,
        size,
        rank=None,
        compression=10,
        num_iters=5,
        num_grad_steps=None,
        trainable_dims=(),
        init="uniform",
        solver="mu",
        contract_params=None,
        verbose=False,
        **kwargs,
    ):
        canonica_polyadic = CanonicalPolyadic(size, rank, compression)

        # set factors initializer
        if f"{init}_" in dir(nn.init):
            init = (RandomTensorInit, {"method": init})

        # set solver
        if solver == "mu":
            solver = GeneralizedMultiplicativeUpdate

        super().__init__(
            canonica_polyadic,
            num_iters=num_iters,
            num_grad_steps=num_grad_steps,
            trainable_dims=trainable_dims,
            init=init,
            solver=solver,
            contract_params=contract_params,
            verbose=verbose,
            **kwargs,
        )


# alias: NTF (Non-negative Tensor Factorization)
NTF = NCPD


class NTD(TF):
    """Non-negative Tucker Decomposition.

        X ≈ (G; U1, ..., UM),
        G, U1, ..., UM ≥ 0
    """

    def __init__(
        self,
        size,
        rank=None,
        compression=10,
        num_iters=5,
        num_grad_steps=None,
        trainable_dims=(),
        init="nnmlsvd",
        solver="mu",
        contract_params=None,
        verbose=False,
        **kwargs,
    ):
        tucker = Tucker(size, rank, compression)

        if f"{init}_" in dir(nn.init):
            init = (RandomTensorInit, {"method": init})
        elif init == "nnmlsvd":
            init = (MLSVDInit, {"nonnegative": True})

        if solver == "mu":
            solver = (
                GeneralizedMultiplicativeUpdate,
                {"contract_params": {"optimize": "auto"}},
            )

        super().__init__(
            tucker,
            num_iters=num_iters,
            num_grad_steps=num_grad_steps,
            trainable_dims=trainable_dims,
            init=init,
            solver=solver,
            contract_params=contract_params,
            verbose=verbose,
            **kwargs,
        )


class NTTD(TF):
    """Non-negative Tensor-Train Decomposition.

        X ≈ train(U1, ..., UM),
        U1, ..., UM ≥ 0
    """

    def __init__(
        self,
        size,
        rank=None,
        compression=10,
        num_iters=5,
        num_grad_steps=None,
        trainable_dims=(),
        init="uniform",
        solver="mu",
        contract_params=None,
        verbose=False,
        **kwargs,
    ):
        tensor_train = TensorTrain(size, rank, compression)

        # set factors initializer
        if f"{init}_" in dir(nn.init):
            init = (RandomTensorInit, {"method": init})

        # set solver
        if solver == "mu":
            solver = GeneralizedMultiplicativeUpdate

        super().__init__(
            tensor_train,
            num_iters=num_iters,
            num_grad_steps=num_grad_steps,
            trainable_dims=trainable_dims,
            init=init,
            solver=solver,
            contract_params=contract_params,
            verbose=verbose,
            **kwargs,
        )


class Nofactorization(nn.Module):
    def __init__(self, *args, **kwarks):
        super().__init__()
        self.flops = {"init": 0, "decompose": 0, "reconstruct": 0}

    def forward(self, x):
        return x

