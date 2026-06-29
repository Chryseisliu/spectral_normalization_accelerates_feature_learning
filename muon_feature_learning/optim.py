from __future__ import annotations

import math

import torch


@torch.no_grad()
def zeroth_power_via_newtonschulz5(
    grad: torch.Tensor,
    steps: int = 5,
    eps: float = 1e-7,
    use_bfloat16: bool = True,
) -> torch.Tensor:
    """Approximate the polar factor U V^T of a 2D matrix.

    This is the Newton-Schulz Muon core popularized by Keller Jordan's
    Muon implementation, with the tuned quintic coefficients used in
    modded-nanogpt.
    """

    if grad.ndim != 2:
        raise ValueError(f"Muon expects a 2D tensor, got shape {tuple(grad.shape)}")

    original_dtype = grad.dtype
    x = grad
    if use_bfloat16 and grad.device.type == "cuda":
        x = x.to(torch.bfloat16)
    else:
        x = x.to(torch.float32)

    x = x / (x.norm() + eps)
    transposed = x.size(0) > x.size(1)
    if transposed:
        x = x.T

    a, b, c = (3.4445, -4.7750, 2.0315)
    for _ in range(steps):
        gram = x @ x.T
        x = a * x + (b * gram + c * gram @ gram) @ x

    if transposed:
        x = x.T
    return x.to(original_dtype)


def _scale_matrix_update(update: torch.Tensor, param: torch.Tensor, update_scale: str) -> torch.Tensor:
    if update_scale == "sqrt_aspect":
        return update * max(1.0, param.size(0) / param.size(1)) ** 0.5
    if update_scale == "none":
        return update
    raise ValueError(f"unknown update_scale={update_scale!r}")


@torch.no_grad()
def _rank_one_polar_factor(x: torch.Tensor, eps: float) -> torch.Tensor:
    col_norms = torch.linalg.vector_norm(x, dim=0)
    col_idx = int(torch.argmax(col_norms).item())
    left = x[:, col_idx]
    left_norm = left.norm()
    if left_norm <= eps:
        return torch.zeros_like(x)
    u = left / left_norm
    right = x.T @ u
    right_norm = right.norm()
    if right_norm <= eps:
        return torch.zeros_like(x)
    v = right / right_norm
    return torch.outer(u, v)


@torch.no_grad()
def spectral_control_update(
    grad: torch.Tensor,
    transform: str,
    *,
    freon_c: float = 0.5,
    trunc_frac: float = 0.05,
    random_high: float = 1.175,
    generator: torch.Generator | None = None,
    assume_rank_one: bool = False,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Apply diagnostic spectral transforms to a matrix gradient.

    These controls intentionally preserve the gradient singular-vector basis.
    They are designed to test whether projected feature-learning gains require
    Muon's Newton-Schulz geometry specifically, or only broader spectral
    normalization/reshaping.
    """

    if grad.ndim != 2:
        raise ValueError(f"spectral controls expect a 2D tensor, got shape {tuple(grad.shape)}")

    original_dtype = grad.dtype
    x = grad.float()
    if assume_rank_one:
        if transform == "trunc_sgd":
            return torch.zeros_like(grad)
        return _rank_one_polar_factor(x, eps).to(original_dtype)

    if transform == "fro_norm":
        return (x / x.norm().clamp_min(eps)).to(original_dtype)

    u, s, vh = torch.linalg.svd(x, full_matrices=False)
    rank = int((s > eps * s.max().clamp_min(eps)).sum().item())
    rank = max(rank, 1)
    target_fro = math.sqrt(rank)

    if transform == "polar":
        vals = torch.zeros_like(s)
        vals[:rank] = 1.0
    elif transform == "rand_spectrum":
        vals = torch.zeros_like(s)
        rand = torch.rand(rank, device=s.device, generator=generator, dtype=s.dtype)
        vals[:rank] = rand * random_high
        vals_norm = vals.norm().clamp_min(eps)
        vals = vals * (target_fro / vals_norm)
    elif transform == "freon":
        vals = torch.zeros_like(s)
        exponent = 1.0 - 2.0 * freon_c
        vals[:rank] = s[:rank].clamp_min(eps).pow(exponent)
        vals_norm = vals.norm().clamp_min(eps)
        vals = vals * (target_fro / vals_norm)
    elif transform == "trunc_sgd":
        vals = s.clone()
        k = max(1, int(math.ceil(trunc_frac * s.numel())))
        vals[:k] = 0.0
        if vals.norm() > eps:
            vals = vals * (s.norm() / vals.norm().clamp_min(eps))
    else:
        raise ValueError(f"unknown spectral transform={transform!r}")

    return ((u * vals.unsqueeze(0)) @ vh).to(original_dtype)


class MuonMatrixState:
    """Momentum state for one matrix parameter updated with Muon."""

    def __init__(
        self,
        param: torch.Tensor,
        lr: float,
        momentum: float = 0.95,
        nesterov: bool = False,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
        update_scale: str = "sqrt_aspect",
    ) -> None:
        if param.ndim != 2:
            raise ValueError("MuonMatrixState only supports one 2D tensor")
        self.param = param
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        self.ns_steps = ns_steps
        self.weight_decay = weight_decay
        self.update_scale = update_scale
        self.buf = torch.zeros_like(param)

    @torch.no_grad()
    def step(self, grad: torch.Tensor) -> None:
        if self.weight_decay:
            self.param.mul_(1.0 - self.lr * self.weight_decay)

        self.buf.mul_(self.momentum).add_(grad)
        update = grad.add(self.buf, alpha=self.momentum) if self.nesterov else self.buf
        update = zeroth_power_via_newtonschulz5(update, steps=self.ns_steps)

        update = _scale_matrix_update(update, self.param, self.update_scale)
        self.param.add_(update, alpha=-self.lr)


class SpectralControlMatrixState:
    """Matrix optimizer state for diagnostic spectral controls."""

    def __init__(
        self,
        param: torch.Tensor,
        lr: float,
        transform: str,
        momentum: float = 0.0,
        nesterov: bool = False,
        weight_decay: float = 0.0,
        update_scale: str = "sqrt_aspect",
        freon_c: float = 0.5,
        trunc_frac: float = 0.05,
        random_high: float = 1.175,
        assume_rank_one: bool = False,
        seed: int = 0,
    ) -> None:
        if param.ndim != 2:
            raise ValueError("SpectralControlMatrixState only supports one 2D tensor")
        self.param = param
        self.lr = lr
        self.transform = transform
        self.momentum = momentum
        self.nesterov = nesterov
        self.weight_decay = weight_decay
        self.update_scale = update_scale
        self.freon_c = freon_c
        self.trunc_frac = trunc_frac
        self.random_high = random_high
        self.assume_rank_one = assume_rank_one
        self.buf = torch.zeros_like(param)
        self.generator = torch.Generator(device=param.device)
        self.generator.manual_seed(seed)

    @torch.no_grad()
    def step(self, grad: torch.Tensor) -> None:
        if self.weight_decay:
            self.param.mul_(1.0 - self.lr * self.weight_decay)

        self.buf.mul_(self.momentum).add_(grad)
        update_source = grad.add(self.buf, alpha=self.momentum) if self.nesterov else self.buf
        update = spectral_control_update(
            update_source,
            self.transform,
            freon_c=self.freon_c,
            trunc_frac=self.trunc_frac,
            random_high=self.random_high,
            generator=self.generator,
            assume_rank_one=self.assume_rank_one,
        )
        update = _scale_matrix_update(update, self.param, self.update_scale)
        self.param.add_(update, alpha=-self.lr)


class VectorSGDState:
    def __init__(self, param: torch.Tensor, lr: float, momentum: float = 0.0, weight_decay: float = 0.0):
        self.param = param
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.buf = torch.zeros_like(param)

    @torch.no_grad()
    def step(self, grad: torch.Tensor) -> None:
        if self.weight_decay:
            grad = grad.add(self.param, alpha=self.weight_decay)
        if self.momentum:
            self.buf.mul_(self.momentum).add_(grad)
            grad = self.buf
        self.param.add_(grad, alpha=-self.lr)


class VectorAdamWState:
    def __init__(
        self,
        param: torch.Tensor,
        lr: float,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        self.param = param
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = torch.zeros_like(param)
        self.v = torch.zeros_like(param)
        self.t = 0

    @torch.no_grad()
    def step(self, grad: torch.Tensor) -> None:
        self.t += 1
        if self.weight_decay:
            self.param.mul_(1.0 - self.lr * self.weight_decay)
        self.m.mul_(self.beta1).add_(grad, alpha=1.0 - self.beta1)
        self.v.mul_(self.beta2).addcmul_(grad, grad, value=1.0 - self.beta2)
        m_hat = self.m / (1.0 - self.beta1**self.t)
        v_hat = self.v / (1.0 - self.beta2**self.t)
        self.param.addcdiv_(m_hat, torch.sqrt(v_hat).add_(self.eps), value=-self.lr)


class VectorRMSPropState:
    def __init__(
        self,
        param: torch.Tensor,
        lr: float,
        alpha: float = 0.99,
        eps: float = 1e-8,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ) -> None:
        self.param = param
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.square_avg = torch.zeros_like(param)
        self.buf = torch.zeros_like(param)

    @torch.no_grad()
    def step(self, grad: torch.Tensor) -> None:
        if self.weight_decay:
            grad = grad.add(self.param, alpha=self.weight_decay)
        self.square_avg.mul_(self.alpha).addcmul_(grad, grad, value=1.0 - self.alpha)
        update = grad / torch.sqrt(self.square_avg).add_(self.eps)
        if self.momentum:
            self.buf.mul_(self.momentum).add_(update)
            update = self.buf
        self.param.add_(update, alpha=-self.lr)
