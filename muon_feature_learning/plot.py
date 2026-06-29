from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def row_group_key(row: dict) -> tuple:
    return (
        row["optimizer"],
        row["grad_mode"],
        row.get("feature_projection", "none"),
        row.get("lr_w", "NA"),
        row.get("lr_v", "NA"),
        row.get("momentum", "NA"),
        row.get("readout_optimizer", "NA"),
        row.get("muon_ns_steps", "NA"),
        row.get("muon_nesterov", "NA"),
        row.get("muon_update_scale", "NA"),
        row.get("rmsprop_alpha", "NA"),
        row.get("rmsprop_momentum", "NA"),
        row.get("freon_c", "NA"),
        row.get("trunc_frac", "NA"),
        row.get("rand_spectrum_high", "NA"),
        row.get("spectral_assume_rank_one", "NA"),
    )


def label_from_group(group: tuple) -> str:
    (
        optimizer,
        _,
        projection,
        lr_w,
        lr_v,
        momentum,
        _,
        _,
        nesterov,
        _,
        rms_alpha,
        rms_momentum,
        freon_c,
        trunc_frac,
        rand_high,
        rank_one,
    ) = group
    label = optimizer if projection == "bap" else f"{optimizer} ({projection})"
    extras = []
    if lr_w not in {"NA", ""} and lr_v not in {"NA", ""}:
        extras.append(f"lr={float(lr_w):g}/{float(lr_v):g}")
    if momentum not in {"NA", ""} and float(momentum) != 0.0:
        extras.append(f"mom={float(momentum):g}")
    if str(nesterov) not in {"NA", "0", "False", "false"}:
        extras.append("nest")
    if optimizer == "rmsprop" and rms_alpha not in {"NA", ""}:
        extras.append(f"alpha={float(rms_alpha):g}")
    if optimizer == "rmsprop" and rms_momentum not in {"NA", ""} and float(rms_momentum) != 0.0:
        extras.append(f"rmsmom={float(rms_momentum):g}")
    if optimizer == "freon" and freon_c not in {"NA", ""}:
        extras.append(f"c={float(freon_c):g}")
    if optimizer == "trunc_sgd" and trunc_frac not in {"NA", ""}:
        extras.append(f"p={float(trunc_frac):g}")
    if optimizer == "rand_spectrum" and rand_high not in {"NA", ""}:
        extras.append(f"hi={float(rand_high):g}")
    if str(rank_one) in {"1", "True", "true"}:
        extras.append("rank1")
    return f"{label} [{', '.join(extras)}]" if extras else label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create basic figures from Muon feature-learning runs.")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--fig-dir", type=Path, required=True)
    parser.add_argument("--beta", type=float, default=None, help="Optional beta for time-series plots.")
    parser.add_argument("--width", type=int, default=None, help="Optional width for time-series plots.")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def plot_final_loss(final_rows: list[dict], fig_dir: Path) -> None:
    grouped = defaultdict(list)
    for row in final_rows:
        grouped[(float(row["beta"]), row_group_key(row))].append(row)

    betas = sorted({key[0] for key in grouped})
    if not betas:
        return

    ncols = min(3, len(betas))
    nrows = int(np.ceil(len(betas) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.6 * nrows), squeeze=False)

    for ax, beta in zip(axes.ravel(), betas):
        for group in sorted({key[1] for key in grouped if key[0] == beta}):
            rows = []
            for key, values in grouped.items():
                if key[0] == beta and key[1] == group:
                    rows.extend(values)
            by_width = defaultdict(list)
            for row in rows:
                by_width[int(row["width"])].append(float(row["loss"]))
            widths = sorted(by_width)
            means = [np.mean(by_width[w]) for w in widths]
            stderrs = [np.std(by_width[w]) / max(1, np.sqrt(len(by_width[w]))) for w in widths]
            ax.errorbar(widths, means, yerr=stderrs, marker="o", capsize=2, label=label_from_group(group))
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title(f"beta={beta:g}")
        ax.set_xlabel("width N")
        ax.set_ylabel("final population loss")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(frameon=False)

    for ax in axes.ravel()[len(betas):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(fig_dir / "final_loss_vs_width.png", dpi=200)
    plt.close(fig)


def plot_exponents(exponent_rows: list[dict], fig_dir: Path) -> None:
    if not exponent_rows:
        return
    grouped = defaultdict(list)
    for row in exponent_rows:
        grouped[label_from_group(row_group_key(row))].append(row)

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for optimizer, rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda r: float(r["beta"]))
        betas = [float(r["beta"]) for r in rows]
        chis = [float(r["chi_width"]) for r in rows]
        ax.plot(betas, chis, marker="o", label=optimizer)
    beta_grid = np.linspace(0.05, 1.8, 200)
    sgd_theory = np.where(beta_grid < 1.0, 2 * beta_grid / (1 + beta_grid), beta_grid)
    ax.plot(beta_grid, sgd_theory, color="black", linestyle="--", linewidth=1, label="SGD theory")
    ax.set_xlabel("source exponent beta")
    ax.set_ylabel("fitted chi from width")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "chi_vs_beta.png", dpi=200)
    plt.close(fig)


def plot_time_series(results_dir: Path, fig_dir: Path, beta: float | None, width: int | None) -> None:
    grouped = defaultdict(list)
    for path in sorted(results_dir.glob("**/metrics.jsonl")):
        rows = read_jsonl(path)
        if not rows:
            continue
        first = rows[0]
        if beta is not None and abs(float(first["beta"]) - beta) > 1e-12:
            continue
        if width is not None and int(first["width"]) != width:
            continue
        grouped[(first["beta"], first["width"], row_group_key(first))].append(rows)

    if not grouped:
        return

    beta0 = beta if beta is not None else sorted({float(k[0]) for k in grouped})[0]
    width0 = width if width is not None else sorted({int(k[1]) for k in grouped})[-1]
    selected = {k: v for k, v in grouped.items() if float(k[0]) == beta0 and int(k[1]) == width0}
    if not selected:
        return

    metrics = [
        ("loss", "population loss"),
        ("spike_m", "spike m(t)"),
        ("hww_gap_q", "H_ww gap"),
        ("r_ii_projected", "projected R_II"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.4))
    for ax, (metric, ylabel) in zip(axes.ravel(), metrics):
        for (_, _, group), runs in sorted(selected.items()):
            by_step = defaultdict(list)
            for run in runs:
                for row in run:
                    fallback = row.get("h_gap_q") if metric == "hww_gap_q" else row.get("r_ii")
                    value = row.get(metric, fallback)
                    if value is not None:
                        by_step[int(row["step"])].append(float(value))
            steps = sorted(by_step)
            means = [np.mean(by_step[s]) for s in steps]
            ax.plot(steps, means, label=label_from_group(group))
        ax.set_xlabel("step")
        ax.set_ylabel(ylabel)
        if metric in {"loss", "r_ii", "r_ii_projected"}:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
    fig.suptitle(f"beta={beta0:g}, width={width0}")
    fig.tight_layout()
    fig.savefig(fig_dir / f"time_series_beta={beta0:g}_width={width0}.png", dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    final_rows = read_csv(args.analysis_dir / "final_runs.csv")
    exponent_rows = read_csv(args.analysis_dir / "scaling_exponents.csv")
    plot_final_loss(final_rows, args.fig_dir)
    plot_exponents(exponent_rows, args.fig_dir)
    plot_time_series(args.results_dir, args.fig_dir, args.beta, args.width)
    print(f"wrote figures to {args.fig_dir}")


if __name__ == "__main__":
    main()
