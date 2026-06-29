from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data import make_spectral_task, sample_batch
from .metrics import metric_dict, population_loss
from .optim import MuonMatrixState, SpectralControlMatrixState, VectorAdamWState, VectorRMSPropState, VectorSGDState


SPECTRAL_CONTROL_TRANSFORMS = {
    "polar": "polar",
    "fro_norm": "fro_norm",
    "rand_spectrum": "rand_spectrum",
    "freon": "freon",
    "trunc_sgd": "trunc_sgd",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a two-layer linear student on a power-law source task.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--optimizer",
        choices=["sgd", "adamw", "rmsprop", "muon", "polar", "fro_norm", "rand_spectrum", "freon", "trunc_sgd"],
        required=True,
    )
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--dim", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=1.25)
    parser.add_argument("--beta", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--grad-mode", choices=["population", "minibatch"], default="population")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--lr-w", type=float, default=0.02, help="Learning rate for hidden matrix W/A.")
    parser.add_argument("--lr-v", type=float, default=0.02, help="Learning rate for readout vector V/w.")
    parser.add_argument("--momentum", type=float, default=0.95)
    parser.add_argument("--readout-optimizer", choices=["sgd", "adamw", "rmsprop"], default="sgd")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--rmsprop-alpha", type=float, default=0.99)
    parser.add_argument("--rmsprop-momentum", type=float, default=0.0)
    parser.add_argument("--muon-ns-steps", type=int, default=5)
    parser.add_argument("--muon-nesterov", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--muon-update-scale", choices=["sqrt_aspect", "none"], default="sqrt_aspect")
    parser.add_argument("--freon-c", type=float, default=0.5)
    parser.add_argument("--trunc-frac", type=float, default=0.05)
    parser.add_argument("--rand-spectrum-high", type=float, default=1.175)
    parser.add_argument(
        "--spectral-assume-rank-one",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use the exact rank-one fast path for spectral controls. Valid for projected population gradients.",
    )
    parser.add_argument(
        "--feature-projection",
        choices=["bap", "none"],
        default="bap",
        help="Apply BAP's fixed P0=A0.T@A0/N projection to feature-matrix gradients.",
    )
    parser.add_argument("--init-scale-w", type=float, default=1.0)
    parser.add_argument("--init-scale-v", type=float, default=1.0)
    parser.add_argument("--normalize-teacher", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--random-signs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bulk-quantile", type=float, default=0.99)
    parser.add_argument("--bulk-exclude", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true", help="Validate setup and write args without training.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing completed run.")
    return parser.parse_args()


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_run_dir(args: argparse.Namespace) -> Path:
    run_name = (
        f"opt={args.optimizer}_beta={args.beta:g}_N={args.width}_D={args.dim}_"
        f"alpha={args.alpha:g}_seed={args.seed}_grad={args.grad_mode}_"
        f"lrw={args.lr_w:g}_lrv={args.lr_v:g}_mom={args.momentum:g}_"
        f"proj={args.feature_projection}_ropt={args.readout_optimizer}_"
        f"ns={args.muon_ns_steps}_nesterov={int(args.muon_nesterov)}_scale={args.muon_update_scale}_"
        f"freonc={args.freon_c:g}_trunc={args.trunc_frac:g}_randhi={args.rand_spectrum_high:g}_"
        f"rmsa={args.rmsprop_alpha:g}_rmsmom={args.rmsprop_momentum:g}"
    )
    return args.out_dir / run_name


def initialize_params(args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator(device=device)
    g.manual_seed(args.seed + 17)
    w = args.init_scale_w * torch.randn(args.width, args.dim, generator=g, device=device, dtype=dtype)
    v = args.init_scale_v * torch.randn(args.width, generator=g, device=device, dtype=dtype)
    return w, v


def population_grads(
    w: torch.Tensor,
    v: torch.Tensor,
    task,
    feature_projection: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    n = w.size(0)
    a = (v @ w) / n
    err = a - task.w_star
    lambda_err = task.lambdas * err
    grad_v = (w @ lambda_err) / n
    grad_w = torch.outer(v, lambda_err) / n
    if feature_projection is not None:
        grad_w = grad_w @ feature_projection
    return grad_w, grad_v


def minibatch_grads(
    w: torch.Tensor,
    v: torch.Tensor,
    task,
    batch_size: int,
    generator: torch.Generator,
    feature_projection: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    w_req = w.detach().requires_grad_(True)
    v_req = v.detach().requires_grad_(True)
    x, y = sample_batch(task, batch_size, generator=generator)
    pred = (x @ w_req.T) @ v_req / w_req.size(0)
    loss = 0.5 * F.mse_loss(pred, y)
    grad_w, grad_v = torch.autograd.grad(loss, (w_req, v_req))
    grad_w = grad_w.detach()
    if feature_projection is not None:
        grad_w = grad_w @ feature_projection
    return grad_w, grad_v.detach()


def main() -> None:
    args = parse_args()
    set_seeds(args.seed)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    device = torch.device(args.device)

    run_dir = make_run_dir(args)
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "summary.json"
    if summary_path.exists() and not args.overwrite:
        print(f"Skipping completed run at {run_dir}. Use --overwrite to rerun.")
        return

    with (run_dir / "args.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args) | {"out_dir": str(args.out_dir), "run_dir": str(run_dir)}, f, indent=2, sort_keys=True)

    task = make_spectral_task(
        dim=args.dim,
        alpha=args.alpha,
        beta=args.beta,
        seed=args.seed + 101,
        device=device,
        dtype=dtype,
        normalize_teacher=args.normalize_teacher,
        random_signs=args.random_signs,
    )
    w, v = initialize_params(args, device, dtype)
    if args.feature_projection == "bap":
        feature_projection = (w.detach().T @ w.detach()) / w.size(0)
    else:
        feature_projection = None
    feature_rate_ratio = args.lr_w / args.lr_v if args.lr_v != 0 else 1.0

    if args.optimizer == "muon":
        w_state = MuonMatrixState(
            w,
            lr=args.lr_w,
            momentum=args.momentum,
            nesterov=args.muon_nesterov,
            ns_steps=args.muon_ns_steps,
            weight_decay=args.weight_decay,
            update_scale=args.muon_update_scale,
        )
    elif args.optimizer == "sgd":
        w_state = VectorSGDState(w, lr=args.lr_w, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == "adamw":
        w_state = VectorAdamWState(w, lr=args.lr_w, weight_decay=args.weight_decay)
    elif args.optimizer == "rmsprop":
        w_state = VectorRMSPropState(
            w,
            lr=args.lr_w,
            alpha=args.rmsprop_alpha,
            momentum=args.rmsprop_momentum,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer in SPECTRAL_CONTROL_TRANSFORMS:
        w_state = SpectralControlMatrixState(
            w,
            lr=args.lr_w,
            transform=SPECTRAL_CONTROL_TRANSFORMS[args.optimizer],
            momentum=args.momentum,
            nesterov=args.muon_nesterov,
            weight_decay=args.weight_decay,
            update_scale=args.muon_update_scale,
            freon_c=args.freon_c,
            trunc_frac=args.trunc_frac,
            random_high=args.rand_spectrum_high,
            assume_rank_one=args.spectral_assume_rank_one,
            seed=args.seed + 4242,
        )
    else:
        raise AssertionError(args.optimizer)

    if args.readout_optimizer == "adamw" or args.optimizer == "adamw":
        v_state = VectorAdamWState(v, lr=args.lr_v, weight_decay=args.weight_decay)
    elif args.readout_optimizer == "rmsprop" or args.optimizer == "rmsprop":
        v_state = VectorRMSPropState(
            v,
            lr=args.lr_v,
            alpha=args.rmsprop_alpha,
            momentum=args.rmsprop_momentum,
            weight_decay=args.weight_decay,
        )
    else:
        v_state = VectorSGDState(v, lr=args.lr_v, momentum=0.0, weight_decay=args.weight_decay)

    if args.dry_run:
        dry = metric_dict(
            0,
            w,
            v,
            task,
            feature_projection=feature_projection,
            feature_rate_ratio=feature_rate_ratio,
            bulk_exclude=args.bulk_exclude,
            bulk_quantile=args.bulk_quantile,
        )
        dry["dry_run"] = 1
        with (run_dir / "metrics.jsonl").open("w", encoding="utf-8") as f:
            f.write(json.dumps(dry, sort_keys=True) + "\n")
        return

    batch_generator = torch.Generator(device=device)
    batch_generator.manual_seed(args.seed + 1009)

    metrics_path = run_dir / "metrics.jsonl"
    with metrics_path.open("w", encoding="utf-8") as f:
        for step in range(args.steps + 1):
            if step % args.log_every == 0 or step == args.steps:
                row = metric_dict(
                    step,
                    w,
                    v,
                    task,
                    feature_projection=feature_projection,
                    feature_rate_ratio=feature_rate_ratio,
                    bulk_exclude=args.bulk_exclude,
                    bulk_quantile=args.bulk_quantile,
                )
                row |= {
                    "optimizer": args.optimizer,
                    "width": args.width,
                    "dim": args.dim,
                    "alpha": args.alpha,
                    "beta": args.beta,
                    "seed": args.seed,
                    "grad_mode": args.grad_mode,
                    "lr_w": args.lr_w,
                    "lr_v": args.lr_v,
                    "momentum": args.momentum,
                    "readout_optimizer": args.readout_optimizer,
                    "feature_projection": args.feature_projection,
                    "muon_ns_steps": args.muon_ns_steps,
                    "muon_nesterov": int(args.muon_nesterov),
                    "muon_update_scale": args.muon_update_scale,
                    "rmsprop_alpha": args.rmsprop_alpha,
                    "rmsprop_momentum": args.rmsprop_momentum,
                    "freon_c": args.freon_c,
                    "trunc_frac": args.trunc_frac,
                    "rand_spectrum_high": args.rand_spectrum_high,
                    "spectral_assume_rank_one": int(args.spectral_assume_rank_one),
                }
                f.write(json.dumps(row, sort_keys=True) + "\n")
                f.flush()

                if not math.isfinite(float(row["loss"])):
                    raise FloatingPointError(f"non-finite loss at step {step}: {row['loss']}")

            if step == args.steps:
                break

            if args.grad_mode == "population":
                grad_w, grad_v = population_grads(w, v, task, feature_projection=feature_projection)
            else:
                grad_w, grad_v = minibatch_grads(
                    w,
                    v,
                    task,
                    args.batch_size,
                    batch_generator,
                    feature_projection=feature_projection,
                )

            w_state.step(grad_w)
            v_state.step(grad_v)

    final = {
        "final_loss": float(population_loss(w, v, task).item()),
        "run_dir": str(run_dir),
        "pid": os.getpid(),
        "optimizer": args.optimizer,
        "beta": args.beta,
        "width": args.width,
        "dim": args.dim,
        "alpha": args.alpha,
        "seed": args.seed,
        "steps": args.steps,
        "grad_mode": args.grad_mode,
        "lr_w": args.lr_w,
        "lr_v": args.lr_v,
        "momentum": args.momentum,
        "readout_optimizer": args.readout_optimizer,
        "feature_projection": args.feature_projection,
        "muon_ns_steps": args.muon_ns_steps,
        "muon_nesterov": int(args.muon_nesterov),
        "muon_update_scale": args.muon_update_scale,
        "rmsprop_alpha": args.rmsprop_alpha,
        "rmsprop_momentum": args.rmsprop_momentum,
        "freon_c": args.freon_c,
        "trunc_frac": args.trunc_frac,
        "rand_spectrum_high": args.rand_spectrum_high,
        "spectral_assume_rank_one": int(args.spectral_assume_rank_one),
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
