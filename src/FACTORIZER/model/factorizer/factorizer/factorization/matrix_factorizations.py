from typing import Sequence
from abc import ABC
import copy
import math
import random

import torch
from torch import nn
import torch.nn.functional as F
import opt_einsum as oe

from ..utils.helpers import (
    as_tuple,
    wrap_class,
    null_context,
    is_wrappable_class,
)
from .operations import t, dot, norm2, relative_error
from .base import MF


################################
# Initializers
################################


class RandomInit(nn.Module):
    def __init__(self, size, rank, method="uniform"):
        super().__init__()
        M, N = size
        method = as_tuple(method)
        if len(method) == 1:
            self.method = (method[0], method[0])
        elif len(method) == 2:
            self.method = method
        else:
            raise ValueError("`method` not valid.")

        init_method = getattr(nn.init, f"{self.method[0]}_")
        self.register_buffer("u0", torch.empty(M, rank))
        init_method(self.u0)

        init_method = getattr(nn.init, f"{self.method[1]}_")
        self.register_buffer("v0", torch.empty(N, rank))
        init_method(self.v0)

        self.flops = 0

    def forward(self, x):
        u_shape = (*x.shape[:-2], *self.u0.shape)
        v_shape = (*x.shape[:-2], *self.v0.shape)

        u = self.u0.expand(u_shape)
        v = self.v0.expand(v_shape)
        return u, v


class FCMInit(nn.Module):
    def __init__(self, size, rank):
        super().__init__()
        self.fcm = FCM(size=size, rank=rank, num_iters=1)
        self.fcm.m = nn.Parameter(torch.tensor([2]))
        self.flops = self.fcm.flops["decompose"]

    def forward(self, x):
        u, v = self.fcm.decompose(x)
        return u, v


class EKMInit(nn.Module):
    def __init__(self, size, rank):
        super().__init__()
        self.ekm = EKM(size=size, rank=rank, num_iters=1)
        self.ekm.alpha = nn.Parameter(torch.tensor([0.1]))
        self.flops = self.ekm.flops["decompose"]

    def forward(self, x):
        u, v = self.ekm.decompose(x)
        return u, v


class SVDInit(nn.Module):
    def __init__(self, size, rank):
        super().__init__()
        (M, N), R = size, rank
        self.svd = SVD(size=size, rank=rank)
        self.flops = self.svd.flops["decompose"] + R + 2 * M * R + 2 * N * R

    def forward(self, x):
        u, s, v = self.svd.decompose(x)
        s = torch.sqrt(s)  # FLOPS = R
        u = torch.einsum("bir, br -> bir", u, s)  # FLOPS = 2MR
        v = torch.einsum("bjr, br -> bjr", v, s)  # FLOPS = 2NR
        return u, v


class NNDSVDInit(nn.Module):
    def __init__(self, size, rank):
        super().__init__()
        self.size = M, N = size
        self.rank = rank
        self.svd = SVD(size, rank)
        self.flops = self.svd.flops["decompose"] + 2 * (4 * rank * (M + N))

    def forward(self, x):
        u, s, v = self.svd.decompose(x)
        s = torch.sqrt(s)
        u = torch.einsum("bir, br -> bir", u, s)
        v = torch.einsum("bjr, br -> bjr", v, s)

        for r in range(self.rank):
            a = u[:, :, r].clone()
            b = v[:, :, r].clone()
            ap = torch.relu(a)
            an = torch.relu(-a)
            bp = torch.relu(b)
            bn = torch.relu(-b)
            abp = norm2(ap) * norm2(bp)
            abn = norm2(an) * norm2(bn)
            batch_mask = abp >= abn
            u[batch_mask, :, r] = ap[batch_mask, :].clone()
            v[batch_mask, :, r] = bp[batch_mask, :].clone()
            u[~batch_mask, :, r] = an[~batch_mask, :].clone()
            v[~batch_mask, :, r] = bn[~batch_mask, :].clone()

        return u, v


################################
# Solvers
################################


class Solver(nn.Module, ABC):
    """An abstract class of a solver."""

    def __init__(self, factor=(0, 1), **kwargs):
        super().__init__()
        self.factor = as_tuple(factor)
        assert set(self.factor).issubset(
            {0, 1}
        ), "`factor` elements must be 0 or 1."
        self.size = kwargs.get("size")  # size and rank used to count flops
        self.rank = kwargs.get("rank")
        self.flops = self.get_flops(self.size, self.rank, self.factor)

    def get_flops_u(self, size, rank):
        raise NotImplementedError(
            f"Subclass {self.__class__.__name__} must implement this method."
        )

    def get_flops_v(self, size, rank):
        return self.get_flops_u(size[::-1], rank)

    def get_flops(self, size=None, rank=None, factor=None):
        size = self.size if size is None else size
        rank = self.rank if rank is None else rank
        factor = self.factor if factor is None else factor
        if None not in (size, rank):
            flops = 0
            for j in self.factor:
                if j == 0:
                    flops += self.get_flops_u(size, rank)
                else:
                    flops += self.get_flops_v(size, rank)
        else:
            flops = None

        return flops

    def update_u(self, x, u, v):
        # x ≈ u @ t(v) --> u = ?
        raise NotImplementedError(
            f"Subclass {self.__class__.__name__} must implement this method."
        )

    def update_v(self, x, u, v):
        # x ≈ u @ t(v) --> v = ?
        return self.update_u(t(x), v, u)

    def forward(self, x, factor_matrices):
        u, v = factor_matrices

        for j in self.factor:
            if j == 0:
                u = self.update_u(x, u, v)
            else:
                v = self.update_v(x, u, v)

        return u, v


class LeastSquares(Solver):
    """Least Squares."""

    def __init__(self, factor=(0, 1), eps=1e-8, project=None, **kwargs):
        super().__init__()
        self.factor = as_tuple(factor)
        assert set(self.factor).issubset(
            {0, 1}
        ), "`factor` elements must be 0 or 1."
        self.eps = eps
        project = nn.Identity if project is None else project
        project = wrap_class(project)
        self.project = project()
        self.size = kwargs.get("size")  # size and rank used to count flops
        self.rank = kwargs.get("rank")
        self.flops = self.get_flops(self.size, self.rank, self.factor)

    def get_flops_u(self, size, rank):
        (M, N), R = size, rank
        flops = math.ceil(2 * N * (R ** 2) * M - (2 / 3) * (R ** 3) * M)
        return flops

    def update_u(self, x, u, v):
        # x ≈ u @ t(v) --> u = ?
        *_, M, N = x.shape
        if M >= N:
            u_new = x @ t(torch.linalg.pinv(v))
        else:
            a, b = x @ v, t(v) @ v
            u_new = torch.linalg.solve(b, t(a))
            u_new = t(u_new)

        return self.project(u_new)


class ProjectedGradient(Solver):
    "Projected gradient descent with line search for linear least squares."

    def __init__(self, factor=(0, 1), project=None, eps=1e-8, **kwargs):
        super().__init__()
        self.factor = as_tuple(factor)
        assert set(self.factor).issubset(
            {0, 1}
        ), "`factor` elements must be 0 or 1."
        self.eps = eps
        project = nn.Identity if project is None else project
        project = wrap_class(project)
        self.project = project()
        self.size = kwargs.get("size")  # size and rank used to count flops
        self.rank = kwargs.get("rank")
        self.flops = self.get_flops(self.size, self.rank, self.factor)

    def get_flops_u(self, size, rank):
        (M, N), R = size, rank
        flops = 2 * M * N * R + 4 * M * R + 2 * N * R + 7 * M * R
        return flops

    def update_u(self, x, u, v):
        # x ≈ u @ t(v) --> u = ?
        a, b = x @ v, t(v) @ v  # FLOPS = 2MNR + 2RNR
        g = a - u @ b  # gradient; FLOPS = MR + 2MRR
        η = (dot(g, g) + self.eps) / (
            dot(g, g @ b) + self.eps
        )  # FLOPS = 2MR + 2MR + 2MRR
        η = η.unsqueeze(-1)
        u_new = self.project(u + η * g)  # FLOPS = 2MR
        return u_new


class CoordinateDescent(Solver):
    "Block coordinate descent update for linear least squares."

    def __init__(self, factor=(0, 1), eps=1e-8, project=None, **kwargs):
        super().__init__()
        self.factor = as_tuple(factor)
        assert set(self.factor).issubset(
            {0, 1}
        ), "`factor` elements must be 0 or 1."
        self.eps = eps
        project = nn.Identity if project is None else project
        project = wrap_class(project)
        self.project = project()
        self.size = kwargs.get("size")  # size and rank used to count flops
        self.rank = kwargs.get("rank")
        self.flops = self.get_flops(self.size, self.rank, self.factor)

    def get_flops_u(self, size, rank):
        (M, N), R = size, rank
        flops = 2 * R * (M * N + N * R + M * R + M)
        return flops

    def update_u(self, x, u, v):
        # x ≈ u @ t(v) --> u = ?
        R = u.shape[-1]
        a, b = x @ v, t(v) @ v  # FLOPS = 2MNR + 2RNR
        if R > 1:
            u_new = u.clone()
            for r in range(R):
                indices = [j for j in range(R) if j != r]
                term1 = a[..., r : (r + 1)].clone()
                term2 = u_new[..., indices].clone()
                term3 = b[..., indices, r : (r + 1)].clone()
                numerator = term1 - term2 @ term3 + self.eps  # FLOPS = 2MR
                denominator = (
                    b[..., r : (r + 1), r : (r + 1)].clone() + self.eps
                )
                u_new[..., r : (r + 1)] = self.project(
                    numerator / denominator
                ).clone()  # FLOPS = 2M
        else:
            numerator = a + self.eps
            denominator = b + self.eps
            u_new = self.project(numerator / denominator)

        return u_new


class MultiplicativeUpdate(Solver):
    def __init__(self, factor=(0, 1), eps=1e-8, **kwargs):
        super().__init__()
        self.factor = as_tuple(factor)
        assert set(self.factor).issubset(
            {0, 1}
        ), "`factor` elements must be 0 or 1."
        self.eps = eps
        self.size = kwargs.get("size")  # size and rank used to count flops
        self.rank = kwargs.get("rank")
        self.flops = self.get_flops(self.size, self.rank, self.factor)

    def get_flops_u(self, size, rank):
        (M, N), R = size, rank
        flops = 2 * R * (M * N + N * R + M * R + 2 * M)
        return flops

    def update_u(self, x, u, v):
        # x ≈ u @ t(v) --> u = ?
        a, b = x @ v, t(v) @ v  # FLOPS = 2MNR + 2RNR
        numerator = u * a + self.eps  # FLOPS = 2MR
        denominator = u @ b + self.eps  # FLOPS = 2MRR + MR
        u_new = numerator / denominator  # FLOPS = MR
        return u_new


class WeightedMultiplicativeUpdate(Solver):
    """Weighted multiplicative update for weighted NMF:

        min || W * (X - U t(V)) ||^2
        s.t. U, V ≥ 0
    """

    def __init__(self, factor=(0, 1), eps=1e-8, **kwargs):
        super().__init__()
        self.factor = as_tuple(factor)
        assert set(self.factor).issubset(
            {0, 1}
        ), "`factor` elements must be 0 or 1."
        self.eps = eps
        self.size = kwargs.get("size")  # size and rank used to count flops
        self.rank = kwargs.get("rank")
        self.flops = self.get_flops(self.size, self.rank, self.factor)

    def get_flops_u(self, size, rank):
        (M, N), R = size, rank
        flops = 2 * (3 * M * N * R + M * N + 2 * M * R)
        return flops

    def update_u(self, x, u, v, w):
        # x ≈ u @ t(v) --> u = ?
        a = (w * x) @ v  # FLOPS = MN + 2MNR
        numerator = u * a + self.eps  # FLOPS = 2MR
        denominator = (
            w * (u @ t(v))
        ) @ v + self.eps  # FLOPS = 2MRN + MN + 2MNR + MR
        u_new = numerator / denominator  # FLOPS = MR
        return u_new

    def update_v(self, x, u, v, w):
        # x ≈ u @ t(v) --> v = ?
        return self.update_u(t(x), v, u, t(w))

    def forward(self, x, factor_matrices):
        u, v = factor_matrices
        weight = torch.ones_like(x) if weight is None else weight

        for j in self.factor:
            if j == 0:
                u = self.update_u(x, u, v, weight)
            else:
                v = self.update_v(x, u, v, weight)

        return u, v


class FastMultiplicativeUpdate(Solver):
    def __init__(self, size, rank, factor=(0, 1), eps=1e-8, **kwargs):
        super().__init__()
        self.size = size
        self.rank = rank
        self.factor = as_tuple(factor)
        assert set(self.factor).issubset(
            {0, 1}
        ), "`factor` elements must be 0 or 1."
        self.eps = eps
        self.expr_u = self.get_expr_u(self.size, self.rank)
        self.expr_v = self.get_expr_v(self.size, self.rank)
        self.flops = self.get_flops(self.size, self.rank, self.factor)

    def get_flops_u(self, size, rank):
        (M, N), R = size, rank
        x_size = (1, M, N)
        u_size = (1, M, R)
        v_size = (1, N, R)
        flops = 0
        _, numerator_info = oe.contract_path(
            "bij, bir, bjr -> bir",
            x_size,
            u_size,
            v_size,
            shapes=True,
            optimize="optimal",
        )
        flops += numerator_info.opt_cost
        _, denominator_info = oe.contract_path(
            "bis, bjs, bjr -> bir",
            u_size,
            v_size,
            v_size,
            shapes=True,
            optimize="optimal",
        )
        flops += denominator_info.opt_cost
        return flops

    def get_flops_v(self, size, rank):
        (M, N), R = size, rank
        x_size = (1, M, N)
        u_size = (1, M, R)
        v_size = (1, N, R)
        flops = 0
        _, numerator_info = oe.contract_path(
            "bij, bir, bjr -> bjr",
            x_size,
            u_size,
            v_size,
            shapes=True,
            optimize="optimal",
        )
        flops += numerator_info.opt_cost
        _, denominator_info = oe.contract_path(
            "bir, bis, bjs -> bjr",
            u_size,
            u_size,
            v_size,
            shapes=True,
            optimize="optimal",
        )
        flops += denominator_info.opt_cost
        return flops

    def get_expr_u(self, size, rank):
        (M, N), R = size, rank
        x_size = (1, M, N)
        u_size = (1, M, R)
        v_size = (1, N, R)
        numerator_expr = oe.contract_expression(
            "bij, bir, bjr -> bir", x_size, u_size, v_size, optimize="optimal"
        )
        denominator_expr = oe.contract_expression(
            "bis, bjs, bjr -> bir", u_size, v_size, v_size, optimize="optimal"
        )
        return numerator_expr, denominator_expr

    def get_expr_v(self, size, rank):
        (M, N), R = size, rank
        x_size = (1, M, N)
        u_size = (1, M, R)
        v_size = (1, N, R)
        numerator_expr = oe.contract_expression(
            "bij, bir, bjr -> bjr", x_size, u_size, v_size, optimize="optimal"
        )
        denominator_expr = oe.contract_expression(
            "bir, bis, bjs -> bjr", u_size, v_size, v_size, optimize="optimal"
        )
        return numerator_expr, denominator_expr

    def update_u(self, x, u, v):
        # x ≈ u @ t(v) --> u = ?
        numerator_expr, denominator_expr = self.expr_u
        numerator = numerator_expr(x, u, v) + self.eps
        denominator = denominator_expr(u, v, v) + self.eps
        u_new = numerator / denominator
        return u_new

    def update_v(self, x, u, v, w):
        # x ≈ u @ t(v) --> v = ?
        numerator_expr, denominator_expr = self.expr_v
        numerator = numerator_expr(x, u, v) + self.eps
        denominator = denominator_expr(u, u, v) + self.eps
        v_new = numerator / denominator
        return v_new


class SemiMultiplicativeUpdate(Solver):
    """Multiplicative update for Semi-NMF:

        min || X - U t(V)) ||^2
        s.t. V ≥ 0
    """

    def __init__(self, factor=(0, 1), eps=1e-8, **kwargs):
        super().__init__()
        self.factor = as_tuple(factor)
        assert set(self.factor).issubset(
            {0, 1}
        ), "`factor` elements must be 0 or 1."
        self.eps = eps
        self.size = kwargs.get("size")  # size and rank used to count flops
        self.rank = kwargs.get("rank")
        self.flops = self.get_flops(self.size, self.rank, self.factor)

    def get_flops_u(self, size, rank):
        (M, N), R = size, rank
        flops = (
            2 * M * N * R
            + 2 * R * N * R
            + 7 * M * R
            + 4 * M * R * R
            + 2 * R * R
        )
        return flops

    def update_u(self, x, u, v):
        # x ≈ u @ t(v) --> u = ?
        a, b = x @ v, t(v) @ v  # FLOPS = 2MNR + 2RNR
        numerator = (
            torch.relu(a) + u @ torch.relu(-b) + self.eps
        )  # FLOPS = MR + (2MRR + RR) + MR
        denominator = (
            torch.relu(-a) + u @ torch.relu(b) + self.eps
        )  # FLOPS = MR + (2MRR + RR) + MR
        u_new = u * torch.sqrt(numerator / denominator)  # FLOPS = MR + MR + MR
        return u_new


class Compose(Solver, Sequence):
    def __init__(self, solvers=None, **kwargs):
        super().__init__()
        if solvers is None:
            solvers = []

        solvers = as_tuple(solvers)
        self.solvers = []
        self.factor = []
        self.size = kwargs.get("size")
        self.rank = kwargs.get("rank")
        self.flops = []
        for solver in solvers:
            solver = wrap_class(solver)
            solver = solver(**kwargs)
            self.solvers.append(solver)
            self.factor.append(getattr(solver, "factor"))
            flops = solver.flops if hasattr(solver, "flops") else None
            self.flops.append(flops)

        self.flops = None if None in self.flops else sum(self.flops)

    def forward(self, x, factor_matrices):
        u, v = factor_matrices
        for solver in self.solvers:
            u, v = solver(x, (u, v))

        return u, v

    def __setitem__(self, idx, solver):
        self.solvers[idx] = solver

    def __getitem__(self, idx):
        return self.solvers[idx]

    def __len__(self):
        return len(self.solvers)

    # def __iter__(self):
    #     return self

    # def __next__(self):  # Python 2: def next(self)
    #     return self.solver[-1]


###################################
# K-means-based factorizers
###################################


class FCM(MF):
    """Fuzzy c-means (FCM), also called fuzzy k-means."""

    def __init__(
        self,
        size,
        rank=None,
        compression=None,
        m=2,
        num_iters=5,
        num_grad_steps=None,
        eps=1e-16,
        seed=42,
        verbose=False,
        **kwargs,
    ):
        self.m = m
        self.eps = eps
        self.seed = seed

        def init(size, rank):
            (M, _), R = size, rank

            def wrapper(x):
                random.seed(self.seed)
                inds = random.sample(range(M), R)
                v = t(x[:, inds, :])
                u = self.get_memberships(x, v)
                return u, v

            wrapper.flops = 0
            return wrapper

        def solver(size, rank):
            def wrapper(x, factors):
                _, v = factors
                u_new = self.get_memberships(x, v)
                v_new = self.get_centers(x, u_new)
                return u_new, v_new

            (M, N), R = size, rank
            wrapper.flops = 4 * M * R * N + 7 * M * R
            return wrapper

        super().__init__(
            size,
            rank=rank,
            compression=compression,
            num_iters=num_iters,
            num_grad_steps=num_grad_steps,
            init=init,
            solver=solver,
            verbose=verbose,
            **kwargs,
        )

    def get_dist(self, x, v):
        x2 = (x ** 2).sum(2, keepdim=True)
        xv = x @ v
        v2 = (v ** 2).sum(1, keepdim=True)
        return torch.relu(x2 - 2 * xv + v2)

    def get_memberships(self, x, v):
        d = self.get_dist(x, v)  # FLOPS = 2MRN
        u = (d + self.eps) ** (1 / (1 - self.m))  # FLOPS = 2MR
        u = (u + self.eps) / (u.sum(2, keepdim=True) + self.eps)  # FLOPS = 2MR
        return u

    def get_centers(self, x, u):
        u = u ** self.m  # FLOPS = MR
        u = (u + self.eps) / (u.sum(1, keepdim=True) + self.eps)  # FLOPS = 2MR
        v = t(x) @ u  # FLOPS = 2MRN
        return v

    def loss(self, x, u, v):
        d = self.get_dist(x, v)
        return torch.mean((u ** self.m) * d)


class EKM(MF):
    """Entropy-based k-means: fuzzy k-means with etropy regularization."""

    def __init__(
        self,
        size,
        rank=None,
        compression=None,
        alpha=0.1,
        num_iters=5,
        num_grad_steps=None,
        eps=1e-16,
        seed=42,
        verbose=False,
        **kwargs,
    ):
        self.eps = eps
        self.seed = seed
        self.verbose = verbose

        def init(size, rank):
            (M, _), R = size, rank

            def wrapper(x):
                random.seed(self.seed)
                inds = random.sample(range(M), R)
                v = t(x[:, inds, :])
                u = self.get_memberships(x, v)
                return u, v

            wrapper.flops = 0
            return wrapper

        def solver(size, rank):
            def wrapper(x, factors):
                _, v = factors
                u_new = self.get_memberships(x, v)
                v_new = self.get_centers(x, u_new)
                return u_new, v_new

            (M, N), R = size, rank
            wrapper.flops = 4 * M * R * N + 7 * M * R
            return wrapper

        super().__init__(
            size,
            rank=rank,
            compression=compression,
            num_iters=num_iters,
            num_grad_steps=num_grad_steps,
            init=init,
            solver=solver,
            verbose=verbose,
            **kwargs,
        )
        (_, N), R = self.size, self.rank
        self.alpha = alpha * N / (1 + math.log(R))

    def get_dist(self, x, v):
        x2 = (x ** 2).sum(2, keepdim=True)
        xv = x @ v
        v2 = (v ** 2).sum(1, keepdim=True)
        return torch.relu(x2 - 2 * xv + v2)

    def get_memberships(self, x, v):
        d = self.get_dist(x, v)  # FLOPS = 2MRN
        u = -d / self.alpha  # FLOPS = 2MR
        u = torch.softmax(u, dim=2)  # FLOPS = 3MR
        return u

    def get_centers(self, x, u):
        u = (u + self.eps) / (u.sum(1, keepdim=True) + self.eps)  # FLOPS = 2MR
        v = t(x) @ u  # FLOPS = 2MRN
        return v

    def loss(self, x, u, v):
        d = self.get_dist(x, v)
        h = torch.where(u > self.eps, u * u.log(), torch.zeros_like(u))
        h = h + (1 / self.rank) * math.log(self.rank)
        return torch.mean(u * d + self.alpha * h)


################################
# Low-rank matrix factorizers
################################


class SVD(nn.Module):
    """Singular Value Dcomposition."""

    def __init__(
        self,
        size,
        rank=None,
        compression=10,
        no_grad=False,
        verbose=False,
        **kwargs,
    ):
        super().__init__()
        self.size = M, N = size
        self.no_grad = no_grad

        assert (rank, compression) != (
            None,
            None,
        ), "'rank' or 'compression' must be specified."

        # degrees of freedom of the input matrix
        df_input = M * N
        # degrees of freedom of the low-rank matrix
        df_lowrank = M + N

        if rank is None:
            self.rank = R = rank = max(
                math.ceil(df_input / (compression * df_lowrank)), 1
            )
        else:
            self.rank = R = rank

        # update compression ratio
        self.compression = df_input / (rank * df_lowrank)

        # reconstruct expression
        reconstruct_eq = "bir, br, bjr -> bij"
        u_size, s_size, v_size = (1, M, R), (1, R), (1, N, R)
        _, self.reconstruct_info = oe.contract_path(
            reconstruct_eq,
            u_size,
            s_size,
            v_size,
            shapes=True,
            optimize="optimal",
        )
        self.reconstruct_expr = oe.contract_expression(
            reconstruct_eq,
            u_size,
            s_size,
            v_size,
            optimize=self.reconstruct_info.path,
        )

        # FLOPs
        self.flops = {"init": 0}
        self.flops["decompose"] = 2 * math.ceil(
            6 * M * N * R + (M + N) * (R ** 2)
        )
        self.flops["reconstruct"] = self.reconstruct_info.opt_cost

        self.verbose = verbose

    def context(self):
        # get context
        if self.no_grad:
            context = torch.no_grad()
        else:
            context = null_context()

        return context

    def decompose(self, x):
        # x: B × M × N
        with self.context():
            torch.manual_seed(42)
            u, s, v = torch.svd_lowrank(x, self.rank)

            if self.verbose:
                loss = self.loss(x, u, s, v)
                print(f"loss = {loss}")

        return u, s, v

    def reconstruct(self, u, s, v):
        return self.reconstruct_expr(u, s, v)

    def loss(self, x, u, s, v):
        return relative_error(x, self.reconstruct(u, s, v))

    def forward(self, x):
        u, s, v = self.decompose(x)
        return self.reconstruct(u, s, v)


class LRMA(MF):
    """Low-Rank Matrix Approximation.

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
        init="normal",
        solver="cd",
        verbose=False,
        **kwargs,
    ):
        # set factors initializer
        init = _wrap_init(init)

        # set solver
        solver = _wrap_solver(solver)

        super().__init__(
            size,
            rank=rank,
            compression=compression,
            num_iters=num_iters,
            num_grad_steps=num_grad_steps,
            init=init,
            solver=solver,
            verbose=verbose,
            **kwargs,
        )


class NMF(MF):
    """Nonnegative Matrix Factorization.

        X ≈ U t(V),
        X, U, V  >= 0
    """

    def __init__(
        self,
        size,
        rank=None,
        compression=10,
        num_iters=5,
        num_grad_steps=None,
        init="uniform",
        solver="hals",
        verbose=False,
        **kwargs,
    ):
        # set factors initializer
        init = _wrap_init(init)

        # set solver
        solver = _wrap_solver(solver)

        super().__init__(
            size,
            rank=rank,
            compression=compression,
            num_iters=num_iters,
            num_grad_steps=num_grad_steps,
            init=init,
            solver=solver,
            verbose=verbose,
            **kwargs,
        )


class GMF(MF):
    """General Matrix Factorization.

        X ≈ U t(V)
        s.t. some contraints
    """

    def __init__(
        self,
        size,
        rank=None,
        compression=10,
        num_iters=5,
        num_grad_steps=None,
        init="uniform-normal",
        solver=("hals-0", "ls-1"),
        verbose=False,
        **kwargs,
    ):
        # set factors initializer
        init = _wrap_init(init)

        # set solver
        solver = _wrap_solver(solver)

        super().__init__(
            size,
            rank=rank,
            compression=compression,
            num_iters=num_iters,
            num_grad_steps=num_grad_steps,
            init=init,
            solver=solver,
            verbose=verbose,
            **kwargs,
        )


class LocalGlobalMF(GMF):
    """Local-Global matrix factorization.

    X = (X1, X2, ..., XJ)
    Xj ≈ Uj t(Vj)
    Uj, Vj ∈ S, Var(U1, ..., Uj) <= ε
    """

    def __init__(
        self,
        size,
        rank=None,
        compression=10,
        alpha=0.01,
        trainable=True,
        num_iters=5,
        num_grad_steps=None,
        init=None,
        solver=None,
        verbose=False,
        **kwargs,
    ):
        super().__init__(
            size,
            rank=rank,
            compression=compression,
            num_iters=num_iters,
            num_grad_steps=num_grad_steps,
            init=init,
            solver=solver,
            verbose=verbose,
            **kwargs,
        )

        # set alpha
        self.alpha = alpha
        gamma = math.log(alpha / (1 - alpha))
        if trainable:
            self.gamma = nn.Parameter(gamma * torch.ones(1))
        else:
            self.gamma = gamma * torch.ones(1)

        self.solver = self.modify_solver(self.solver)

    def modify_get_flops_u(self, get_flops_u):
        def wrapper(size, rank):
            (M, N), R = size, rank
            return get_flops_u((M, (N + R)), R)

        return wrapper

    def modify_update_u(self, update_u):
        def wrapper(x, u, v, *args, **kwargs):
            J, R = u.shape[-3], u.shape[-1]
            λ = torch.exp(self.gamma)
            c = u.mean(-3, keepdim=True).repeat_interleave(J, -3)
            x = torch.cat((x, (λ ** 0.5) * c), dim=-1)
            v = torch.cat((v, (λ ** 0.5) * self.eye_like(v, R)), dim=-2)
            return update_u(x, u, v, *args, **kwargs)

        return wrapper

    def eye_like(self, x: torch.Tensor, n: int) -> torch.Tensor:
        eye = torch.eye(n, n, dtype=x.dtype, device=x.device)
        out = eye.expand((*x.shape[:2], n, n))
        return out

    def modify_solver(self, solver):
        new_solver = copy.deepcopy(solver)
        if isinstance(new_solver, Sequence):
            num_solvers = len(solver)
            for j in range(num_solvers):
                if hasattr(solver[j], "update_u"):
                    new_solver[j].update_u = self.modify_update_u(
                        new_solver[j].update_u
                    )
                    new_solver[j].get_flops_u = self.modify_get_flops_u(
                        new_solver[j].get_flops_u
                    )
                    new_solver[j].flops = new_solver[j].get_flops()
        else:
            if hasattr(solver, "update_u"):
                new_solver.update_u = self.modify_update_u(new_solver.update_u)
                new_solver.get_flops_u = self.modify_get_flops_u(
                    new_solver.get_flops_u
                )
                new_solver.flops = new_solver.get_flops()

        return new_solver


def _dispatch(obj, dispatch_map):
    out = dispatch_map.get(obj, obj) if isinstance(obj, str) else obj
    return out


INIT_DISPATCH_MAP = {
    "uniform": (RandomInit, {"method": "uniform"}),
    "normal": (RandomInit, {"method": "normal"}),
    "normal-uniform": (RandomInit, {"method": ("normal", "uniform")}),
    "uniform-normal": (RandomInit, {"method": ("uniform", "normal")}),
    "fcm": FCMInit,
    "ekm": EKMInit,
    "svd": SVDInit,
    "nndsvd": NNDSVDInit,
}

SOLVER_DISPATCH_MAP = {
    "mu": MultiplicativeUpdate,
    "mu-0": (MultiplicativeUpdate, {"factor": 0}),
    "mu-1": (MultiplicativeUpdate, {"factor": 1}),
    "wmu": WeightedMultiplicativeUpdate,
    "wmu-0": (MultiplicativeUpdate, {"factor": 0}),
    "wmu-1": (MultiplicativeUpdate, {"factor": 1}),
    "smu": SemiMultiplicativeUpdate,
    "smu-0": (SemiMultiplicativeUpdate, {"factor": 0}),
    "smu-1": (SemiMultiplicativeUpdate, {"factor": 1}),
    "cd": CoordinateDescent,
    "cd-0": (CoordinateDescent, {"factor": 0}),
    "cd-1": (CoordinateDescent, {"factor": 1}),
    "nncd": (CoordinateDescent, {"project": nn.ReLU}),
    "nncd-0": (CoordinateDescent, {"factor": 0, "project": nn.ReLU}),
    "nncd-1": (CoordinateDescent, {"factor": 1, "project": nn.ReLU}),
    "hals": (CoordinateDescent, {"project": nn.ReLU}),
    "hals-0": (CoordinateDescent, {"factor": 0, "project": nn.ReLU}),
    "hals-1": (CoordinateDescent, {"factor": 1, "project": nn.ReLU}),
    "ls": LeastSquares,
    "ls-0": (LeastSquares, {"factor": 0}),
    "ls-1": (LeastSquares, {"factor": 1}),
    "nnls": (LeastSquares, {"project": nn.ReLU}),
    "nnls-0": (LeastSquares, {"factor": 0, "project": nn.ReLU}),
    "nnls-1": (LeastSquares, {"factor": 1, "project": nn.ReLU}),
}


def _wrap_init(obj):
    return _dispatch(obj, INIT_DISPATCH_MAP)


def _wrap_solver(obj):
    if is_wrappable_class(obj):
        return obj
    elif isinstance(obj, str):
        return _dispatch(obj, SOLVER_DISPATCH_MAP)
    elif isinstance(obj, Sequence):
        out = []
        for x in obj:
            if is_wrappable_class(x):
                out.append(x)
            elif isinstance(x, str):
                out.append(_dispatch(x, SOLVER_DISPATCH_MAP))
            else:
                raise ValueError
        return (Compose, {"solvers": out})
    else:
        raise ValueError
