from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SpectralTask:
    """Power-law Gaussian covariates with a source-condition teacher."""

    dim: int
    alpha: float
    beta: float
    lambdas: torch.Tensor
    w_star: torch.Tensor

    @property
    def teacher_norm_lambda(self) -> torch.Tensor:
        return torch.sqrt(torch.sum(self.lambdas * self.w_star.square()).clamp_min(1e-30))


def make_spectral_task(
    dim: int,
    alpha: float,
    beta: float,
    seed: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    normalize_teacher: bool = True,
    random_signs: bool = True,
) -> SpectralTask:
    """Create a task with lambda_k = k^-alpha and source exponent beta.

    The source condition is

        sum_{l > k} lambda_l (w_l*)^2 ~ k^{-alpha beta}.

    Since the tail of sum l^{-p} scales as k^{-(p - 1)}, this implies

        lambda_k (w_k*)^2 ~ k^{-(alpha beta + 1)}
        w_k* ~ k^{-[1 + alpha(beta - 1)]/2}.
    """

    device = torch.device(device)
    g = torch.Generator(device=device)
    g.manual_seed(seed)

    k = torch.arange(1, dim + 1, device=device, dtype=dtype)
    lambdas = k.pow(-alpha)

    coeff_scale = k.pow(-(1.0 + alpha * (beta - 1.0)) / 2.0)
    if random_signs:
        signs = torch.randint(0, 2, (dim,), generator=g, device=device).to(dtype)
        signs = signs.mul(2.0).sub(1.0)
        w_star = coeff_scale * signs
    else:
        w_star = coeff_scale

    if normalize_teacher:
        norm = torch.sqrt(torch.sum(lambdas * w_star.square()).clamp_min(1e-30))
        w_star = w_star / norm

    return SpectralTask(
        dim=dim,
        alpha=alpha,
        beta=beta,
        lambdas=lambdas,
        w_star=w_star,
    )


def sample_batch(
    task: SpectralTask,
    batch_size: int,
    seed: int | None = None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample x ~ N(0, Lambda), y = x dot w_star."""

    if generator is None:
        generator = torch.Generator(device=task.lambdas.device)
        if seed is not None:
            generator.manual_seed(seed)

    x = torch.randn(
        batch_size,
        task.dim,
        generator=generator,
        device=task.lambdas.device,
        dtype=task.lambdas.dtype,
    )
    x = x * torch.sqrt(task.lambdas).unsqueeze(0)
    y = x @ task.w_star
    return x, y
