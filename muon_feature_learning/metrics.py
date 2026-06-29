from __future__ import annotations

import math

import torch

from .data import SpectralTask


@torch.no_grad()
def effective_weight(w: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return (v @ w) / w.size(0)


@torch.no_grad()
def population_loss(w: torch.Tensor, v: torch.Tensor, task: SpectralTask) -> torch.Tensor:
    residual = task.w_star - effective_weight(w, v)
    return 0.5 * torch.sum(task.lambdas * residual.square())


@torch.no_grad()
def metric_dict(
    step: int,
    w: torch.Tensor,
    v: torch.Tensor,
    task: SpectralTask,
    feature_projection: torch.Tensor | None = None,
    feature_rate_ratio: float = 1.0,
    bulk_exclude: int = 1,
    bulk_quantile: float = 0.99,
) -> dict[str, float | int]:
    n, d = w.shape
    gamma = n / d
    a = effective_weight(w, v)
    residual = task.w_star - a
    lambda_residual = task.lambdas * residual
    teacher_norm = task.teacher_norm_lambda

    loss = 0.5 * torch.sum(task.lambdas * residual.square())
    spike = torch.dot(a, task.w_star) / teacher_norm.clamp_min(1e-30)
    residual_lambda_norm = torch.linalg.vector_norm(lambda_residual)
    if feature_projection is not None:
        projected_lambda_residual = feature_projection.float() @ lambda_residual.float()
        projected_teacher = feature_projection.float() @ task.w_star.float()
    else:
        projected_lambda_residual = lambda_residual.float()
        projected_teacher = task.w_star.float()
    projected_lambda_residual_norm = torch.linalg.vector_norm(projected_lambda_residual)

    svals = torch.linalg.svdvals(w.float())
    svals_sorted, _ = torch.sort(svals, descending=True)
    if svals_sorted.numel() > bulk_exclude:
        bulk_s = svals_sorted[bulk_exclude:]
    else:
        bulk_s = svals_sorted
    sv_bulk_edge = torch.quantile(bulk_s, bulk_quantile)
    sv_gap = svals_sorted[0] - sv_bulk_edge

    weighted_w = w.float() * torch.sqrt(task.lambdas.float()).unsqueeze(0)
    h_proxy = (weighted_w @ weighted_w.T) / n
    h_eigs = torch.linalg.eigvalsh(h_proxy)
    h_eigs_sorted, _ = torch.sort(h_eigs, descending=True)
    if h_eigs_sorted.numel() > bulk_exclude:
        bulk_h = h_eigs_sorted[bulk_exclude:]
    else:
        bulk_h = h_eigs_sorted
    h_bulk_edge = torch.quantile(bulk_h, bulk_quantile)
    h_gap = h_eigs_sorted[0] - h_bulk_edge
    hww_top = h_eigs_sorted[0] / n
    hww_bulk_edge = h_bulk_edge / n
    hww_gap = h_gap / n

    trace_a = torch.sum(w.float().square()) / n
    trace_b = torch.sum(w.float().square() * task.lambdas.float().unsqueeze(0)) / n
    hww_trace_over_n = trace_b / (n * n)
    haa_trace_over_n = (v.float().square().sum() * task.lambdas.float().sum()) / (n**3)

    denom_r_ii = v.float().norm() * residual_lambda_norm.float()
    r_ii = 1.0 / denom_r_ii.clamp_min(1e-30)
    denom_r_ii_projected = feature_rate_ratio * v.float().norm() * projected_lambda_residual_norm
    r_ii_projected = 1.0 / denom_r_ii_projected.clamp_min(1e-30)
    projected_alignment_cos = torch.dot(projected_lambda_residual, task.w_star.float()) / (
        projected_lambda_residual_norm * task.w_star.float().norm()
    ).clamp_min(1e-30)

    sigma_wt_v = task.lambdas.float() * (w.float().T @ v.float())
    denom_r_i = torch.linalg.vector_norm(sigma_wt_v)
    r_i = 1.0 / denom_r_i.clamp_min(1e-30)

    target_unit = task.w_star / torch.linalg.vector_norm(task.w_star).clamp_min(1e-30)
    top_right = torch.linalg.svd(w.float(), full_matrices=False).Vh[0, :]
    top_align_abs = torch.abs(torch.dot(top_right, target_unit.float()))

    h_edge_isotropic = float(task.lambdas.max().item() * (1.0 + math.sqrt(gamma)) ** 2)

    return {
        "step": int(step),
        "loss": float(loss.item()),
        "spike_m": float(spike.item()),
        "residual_lambda_norm": float(residual_lambda_norm.item()),
        "projected_lambda_residual_norm": float(projected_lambda_residual_norm.item()),
        "projected_teacher_norm": float(torch.linalg.vector_norm(projected_teacher).item()),
        "projected_residual_teacher_cos": float(projected_alignment_cos.item()),
        "feature_rate_ratio": float(feature_rate_ratio),
        "v_norm": float(v.norm().item()),
        "w_fro_over_n": float(trace_a.item()),
        "trace_w_lambda_wt_over_n": float(trace_b.item()),
        "hww_trace_over_n": float(hww_trace_over_n.item()),
        "haa_trace_over_n": float(haa_trace_over_n.item()),
        "top_singular_value": float(svals_sorted[0].item()),
        "sv_bulk_edge_q": float(sv_bulk_edge.item()),
        "sv_gap_q": float(sv_gap.item()),
        "top_h_proxy_eig": float(h_eigs_sorted[0].item()),
        "h_bulk_edge_q": float(h_bulk_edge.item()),
        "h_gap_q": float(h_gap.item()),
        "h_edge_isotropic": h_edge_isotropic,
        "hww_top_eig": float(hww_top.item()),
        "hww_bulk_edge_q": float(hww_bulk_edge.item()),
        "hww_gap_q": float(hww_gap.item()),
        "hww_edge_isotropic": h_edge_isotropic / n,
        "top_right_teacher_align_abs": float(top_align_abs.item()),
        "r_i": float(r_i.item()),
        "r_ii": float(r_ii.item()),
        "r_ii_projected": float(r_ii_projected.item()),
    }
