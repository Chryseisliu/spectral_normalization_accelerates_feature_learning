from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


KEYS = [
    "optimizer",
    "beta",
    "width",
    "dim",
    "alpha",
    "seed",
    "grad_mode",
    "lr_w",
    "lr_v",
    "momentum",
    "feature_projection",
    "readout_optimizer",
    "muon_ns_steps",
    "muon_nesterov",
    "muon_update_scale",
    "rmsprop_alpha",
    "rmsprop_momentum",
    "freon_c",
    "trunc_frac",
    "rand_spectrum_high",
    "spectral_assume_rank_one",
]

GROUP_KEYS = [
    "optimizer",
    "beta",
    "grad_mode",
    "feature_projection",
    "lr_w",
    "lr_v",
    "momentum",
    "readout_optimizer",
    "muon_ns_steps",
    "muon_nesterov",
    "muon_update_scale",
    "rmsprop_alpha",
    "rmsprop_momentum",
    "freon_c",
    "trunc_frac",
    "rand_spectrum_high",
    "spectral_assume_rank_one",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate Muon feature-learning runs and fit exponents.")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--bbp-gap-key", default="hww_gap_q", choices=["hww_gap_q", "h_gap_q", "sv_gap_q"])
    parser.add_argument("--bbp-gap-threshold", type=float, default=0.0)
    parser.add_argument("--align-threshold", type=float, default=0.1)
    parser.add_argument("--min-widths", type=int, default=3)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def fit_power_law(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    xs_array = np.asarray(xs, dtype=float)
    ys_array = np.asarray(ys, dtype=float)
    ok = np.isfinite(xs_array) & np.isfinite(ys_array) & (xs_array > 0) & (ys_array > 0)
    if ok.sum() < 2:
        return float("nan"), float("nan"), float("nan")
    x = np.log(xs_array[ok])
    y = np.log(ys_array[ok])
    slope, intercept = np.polyfit(x, y, deg=1)
    y_hat = slope * x + intercept
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(slope), float(intercept), r2


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    final_rows = []
    time_rows = []
    for metrics_path in sorted(args.results_dir.glob("**/metrics.jsonl")):
        rows = read_jsonl(metrics_path)
        if not rows:
            continue
        final = rows[-1]
        bbp_step = None
        align_step = None
        for row in rows:
            gap_value = row.get(args.bbp_gap_key, row.get("h_gap_q", 0.0))
            if bbp_step is None and gap_value > args.bbp_gap_threshold:
                bbp_step = row["step"]
            if align_step is None and row.get("top_right_teacher_align_abs", 0.0) > args.align_threshold:
                align_step = row["step"]
        final["bbp_step"] = -1 if bbp_step is None else bbp_step
        final["align_step"] = -1 if align_step is None else align_step
        final["source"] = str(metrics_path)
        final_rows.append(final)
        time_rows.extend(rows)

    aggregate_path = args.out_dir / "final_runs.csv"
    if final_rows:
        fieldnames = sorted(set().union(*(row.keys() for row in final_rows)))
        with aggregate_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(final_rows)

    grouped = defaultdict(list)
    for row in final_rows:
        grouped[tuple(row.get(key, "NA") for key in GROUP_KEYS)].append(row)

    exponent_rows = []
    for group_key, rows in sorted(grouped.items(), key=lambda kv: (float(kv[0][1]), kv[0][0], kv[0][2], kv[0][3])):
        group = dict(zip(GROUP_KEYS, group_key))
        by_width = defaultdict(list)
        for row in rows:
            by_width[int(row["width"])].append(row)

        widths = []
        mean_loss = []
        mean_gap = []
        mean_hww_trace = []
        mean_haa_trace = []
        mean_bbp = []
        for width, width_rows in sorted(by_width.items()):
            widths.append(width)
            mean_loss.append(float(np.mean([r["loss"] for r in width_rows])))
            mean_gap.append(float(np.mean([max(r.get(args.bbp_gap_key, r.get("h_gap_q", 1e-30)), 1e-30) for r in width_rows])))
            mean_hww_trace.append(float(np.mean([max(r.get("hww_trace_over_n", 1e-30), 1e-30) for r in width_rows])))
            mean_haa_trace.append(float(np.mean([max(r.get("haa_trace_over_n", 1e-30), 1e-30) for r in width_rows])))
            valid_bbp = [r["bbp_step"] for r in width_rows if r["bbp_step"] >= 0]
            mean_bbp.append(float(np.mean(valid_bbp)) if valid_bbp else float("nan"))

        if len(widths) < args.min_widths:
            continue

        loss_slope, loss_intercept, loss_r2 = fit_power_law(widths, mean_loss)
        gap_slope, gap_intercept, gap_r2 = fit_power_law(widths, mean_gap)
        hww_trace_slope, hww_trace_intercept, hww_trace_r2 = fit_power_law(widths, mean_hww_trace)
        haa_trace_slope, haa_trace_intercept, haa_trace_r2 = fit_power_law(widths, mean_haa_trace)

        exponent_rows.append(
            {
                **group,
                "num_widths": len(widths),
                "num_runs": len(rows),
                "chi_width": -loss_slope,
                "loss_slope": loss_slope,
                "loss_r2": loss_r2,
                "delta_gap": -gap_slope,
                "gap_slope": gap_slope,
                "gap_r2": gap_r2,
                "tau_hww_trace": -hww_trace_slope,
                "hww_trace_slope": hww_trace_slope,
                "hww_trace_r2": hww_trace_r2,
                "tau_haa_trace": -haa_trace_slope,
                "haa_trace_slope": haa_trace_slope,
                "haa_trace_r2": haa_trace_r2,
                "widths": " ".join(map(str, widths)),
                "mean_losses": " ".join(f"{x:.8e}" for x in mean_loss),
                "mean_bbp_steps": " ".join("nan" if np.isnan(x) else f"{x:.2f}" for x in mean_bbp),
            }
        )

    exponent_path = args.out_dir / "scaling_exponents.csv"
    if exponent_rows:
        with exponent_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(exponent_rows[0].keys()))
            writer.writeheader()
            writer.writerows(exponent_rows)

    print(f"wrote {aggregate_path}")
    print(f"wrote {exponent_path}")


if __name__ == "__main__":
    main()
