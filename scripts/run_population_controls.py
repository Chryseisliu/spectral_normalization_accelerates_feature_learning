from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path


PYTHON_EXE = os.environ.get("PYTHON_EXE", sys.executable)
NS5_RANK_ONE_SCALAR = 0.696436

CONTROL_LRS = {
    "sgd": (0.04, 0.04),
    "trunc_sgd": (0.04, 0.04),
    "muon": (0.04, 0.04),
    "polar": (0.04 * NS5_RANK_ONE_SCALAR, 0.04),
    "fro_norm": (0.04 * NS5_RANK_ONE_SCALAR, 0.04),
    "rand_spectrum": (0.04 * NS5_RANK_ONE_SCALAR, 0.04),
    "freon": (0.04 * NS5_RANK_ONE_SCALAR, 0.04),
}


def parse_csv(value: str, cast):
    return [cast(x) for x in value.split(",") if x]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the projected-population spectral-control grid.")
    parser.add_argument("--results-dir", default="results/control_allbeta_10k")
    parser.add_argument("--analysis-dir", default="analysis/control_allbeta_10k")
    parser.add_argument("--log-dir", default="logs/control_allbeta_10k")
    parser.add_argument("--betas", default="0.2,0.5,0.8,1.2,1.5")
    parser.add_argument("--widths", default="32,64,128,256,512,1024,2048")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--optimizers", default="sgd,trunc_sgd,muon,polar,fro_norm,rand_spectrum,freon")
    parser.add_argument("--dim", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=1.25)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--analysis-min-widths", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def expected_runs(args: argparse.Namespace) -> list[dict]:
    runs = []
    for beta in parse_csv(args.betas, float):
        for width in parse_csv(args.widths, int):
            for seed in parse_csv(args.seeds, int):
                for optimizer in parse_csv(args.optimizers, str):
                    if optimizer not in CONTROL_LRS:
                        raise ValueError(f"unknown control optimizer: {optimizer}")
                    lr_w, lr_v = CONTROL_LRS[optimizer]
                    runs.append(
                        {
                            "optimizer": optimizer,
                            "beta": beta,
                            "width": width,
                            "seed": seed,
                            "lr_w": lr_w,
                            "lr_v": lr_v,
                            "spectral_assume_rank_one": optimizer in {"polar", "fro_norm", "rand_spectrum", "freon", "trunc_sgd"},
                        }
                    )
    return runs


def run_key(row: dict) -> tuple:
    return (
        row["optimizer"],
        float(row["beta"]),
        int(row["width"]),
        int(row["seed"]),
        float(row["lr_w"]),
        float(row["lr_v"]),
        int(row.get("spectral_assume_rank_one", False)),
    )


def completed_keys(results_dir: Path) -> set[tuple]:
    keys = set()
    for path in results_dir.glob("**/summary.json"):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
            loss = float(row["final_loss"])
            if math.isfinite(loss):
                keys.add(run_key(row))
        except Exception:
            continue
    return keys


def build_cmd(args: argparse.Namespace, run: dict) -> list[str]:
    cmd = [
        PYTHON_EXE,
        "-m",
        "muon_feature_learning.train",
        "--out-dir",
        args.results_dir,
        "--optimizer",
        run["optimizer"],
        "--beta",
        str(run["beta"]),
        "--width",
        str(run["width"]),
        "--dim",
        str(args.dim),
        "--alpha",
        str(args.alpha),
        "--seed",
        str(run["seed"]),
        "--steps",
        str(args.steps),
        "--batch-size",
        str(args.batch_size),
        "--log-every",
        str(args.log_every),
        "--grad-mode",
        "population",
        "--lr-w",
        str(run["lr_w"]),
        "--lr-v",
        str(run["lr_v"]),
        "--momentum",
        "0.0",
        "--feature-projection",
        "bap",
        "--readout-optimizer",
        "sgd",
        "--muon-ns-steps",
        "5",
        "--no-muon-nesterov",
        "--muon-update-scale",
        "sqrt_aspect",
        "--freon-c",
        "0.5",
        "--trunc-frac",
        "0.05",
        "--rand-spectrum-high",
        "1.175",
        "--spectral-assume-rank-one" if run["spectral_assume_rank_one"] else "--no-spectral-assume-rank-one",
        "--device",
        args.device,
    ]
    if args.overwrite:
        cmd.append("--overwrite")
    return cmd


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    analysis_dir = Path(args.analysis_dir)
    log_dir = Path(args.log_dir)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    runs = expected_runs(args)
    (analysis_dir / "manifest.json").write_text(json.dumps(runs, indent=2), encoding="utf-8")
    done = completed_keys(results_dir) if not args.overwrite else set()
    todo = [run for run in runs if run_key(run) not in done]
    print(f"expected={len(runs)} completed={len(done)} todo={len(todo)}", flush=True)

    active = []
    failed = []
    next_idx = 0
    start = time.time()
    while next_idx < len(todo) or active:
        while next_idx < len(todo) and len(active) < args.max_workers:
            run = todo[next_idx]
            gpu = next_idx % max(1, args.max_workers)
            env = os.environ.copy()
            if args.device.startswith("cuda"):
                env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            log_path = log_dir / f"{next_idx:04d}_{run['optimizer']}_b{run['beta']}_N{run['width']}_s{run['seed']}.log"
            log_file = log_path.open("w", encoding="utf-8")
            cmd = build_cmd(args, run)
            print(
                f"[{next_idx + 1}/{len(todo)}] {run['optimizer']} "
                f"beta={run['beta']} N={run['width']} seed={run['seed']} "
                f"lr={run['lr_w']:.8g}/{run['lr_v']:.8g}",
                flush=True,
            )
            proc = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)
            active.append((next_idx, proc, log_file, log_path, run))
            next_idx += 1

        time.sleep(args.poll_seconds)
        still = []
        for idx, proc, log_file, log_path, run in active:
            rc = proc.poll()
            if rc is None:
                still.append((idx, proc, log_file, log_path, run))
            else:
                log_file.close()
                if rc != 0:
                    failed.append({"idx": idx, "rc": rc, "log": str(log_path), **run})
                    print(f"FAILED idx={idx} rc={rc} log={log_path}", flush=True)
        active = still

    if failed:
        (analysis_dir / "failed.json").write_text(json.dumps(failed, indent=2), encoding="utf-8")
        raise SystemExit(f"{len(failed)} runs failed; see {analysis_dir / 'failed.json'}")

    print(f"runs done in {time.time() - start:.1f}s", flush=True)
    subprocess.run(
        [
            PYTHON_EXE,
            "-m",
            "muon_feature_learning.analysis",
            "--results-dir",
            args.results_dir,
            "--out-dir",
            args.analysis_dir,
            "--min-widths",
            str(args.analysis_min_widths),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
