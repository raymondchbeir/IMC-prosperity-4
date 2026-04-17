#!/usr/bin/env python3
"""
Phase 1 Bayesian Optimization setup for the merged EMERALDS + TOMATOES strategy.

What this tunes:
- TOM_FAIR_ALPHA
- TOM_PRESSURE_BETA
- TOM_TAKE_EDGE
- TOM_PASSIVE_EDGE

How it works:
1. Reads a base strategy file.
2. Patches the 4 TOMATOES parameters for each trial.
3. Writes a temporary strategy file.
4. Runs your backtest command.
5. Extracts metrics from stdout/stderr or an optional JSON file.
6. Optimizes a weighted score.

Recommended use:
- Keep EMERALDS fixed.
- Use multiple backtest seeds / rounds if your environment supports that.
- Prefer score = pnl + small_sharpe_bonus, not raw pnl only.

Example:
python bo_phase1_setup.py \
  --base-strategy /mnt/data/57265_merged_best_emeralds_tomatoes.py \
  --backtest-cmd "python run_backtest.py --strategy {strategy_path}" \
  --n-trials 40 \
  --study-name tomatoes_phase1

If your backtester writes metrics to JSON:
python bo_phase1_setup.py \
  --base-strategy /mnt/data/57265_merged_best_emeralds_tomatoes.py \
  --backtest-cmd "python run_backtest.py --strategy {strategy_path} --out {metrics_path}" \
  --metrics-mode json_file

Supported metrics inputs:
- json_stdout: backtest prints JSON like {"pnl": 1234, "sharpe": 1.2}
- json_file: backtest writes JSON to metrics_path
- regex_stdout: backtest prints text containing things like "PnL: 1234" / "Sharpe: 1.2"

Notes:
- This script uses Optuna's TPE sampler, which is a practical BO-style optimizer.
- If your simulator is deterministic, this setup is fine.
- If your simulator has randomness, average multiple runs per trial.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shlex
import statistics
import subprocess
import sys
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import optuna
except Exception as e:
    print("Optuna is required for this setup. Install it with: pip install optuna", file=sys.stderr)
    raise

PARAM_BOUNDS = {
    "TOM_FAIR_ALPHA": (0.05, 0.40),
    "TOM_PRESSURE_BETA": (0.30, 1.50),
    "TOM_TAKE_EDGE": (0.50, 2.50),
    "TOM_PASSIVE_EDGE": (0, 2),
}

PARAM_PATTERNS = {
    "TOM_FAIR_ALPHA": re.compile(r"^(\s*TOM_FAIR_ALPHA\s*=\s*)([^\n#]+)(.*)$", re.MULTILINE),
    "TOM_PRESSURE_BETA": re.compile(r"^(\s*TOM_PRESSURE_BETA\s*=\s*)([^\n#]+)(.*)$", re.MULTILINE),
    "TOM_TAKE_EDGE": re.compile(r"^(\s*TOM_TAKE_EDGE\s*=\s*)([^\n#]+)(.*)$", re.MULTILINE),
    "TOM_PASSIVE_EDGE": re.compile(r"^(\s*TOM_PASSIVE_EDGE\s*=\s*)([^\n#]+)(.*)$", re.MULTILINE),
}

PNL_PATTERNS = [
    re.compile(r'"pnl"\s*:\s*(-?\d+(?:\.\d+)?)', re.IGNORECASE),
    re.compile(r"\bPnL\b\s*[:=]\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE),
    re.compile(r"\bprofit\b\s*[:=]\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE),
]

SHARPE_PATTERNS = [
    re.compile(r'"sharpe"\s*:\s*(-?\d+(?:\.\d+)?)', re.IGNORECASE),
    re.compile(r"\bSharpe\b\s*[:=]\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE),
]

INV_PATTERNS = [
    re.compile(r'"inventory_penalty"\s*:\s*(-?\d+(?:\.\d+)?)', re.IGNORECASE),
    re.compile(r'"inventory_variance"\s*:\s*(-?\d+(?:\.\d+)?)', re.IGNORECASE),
    re.compile(r"\binventory(?:_penalty|_variance)?\b\s*[:=]\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE),
]


@dataclass
class TrialMetrics:
    pnl: float
    sharpe: float = 0.0
    inventory_penalty: float = 0.0
    raw_output: str = ""


def patch_strategy_text(text: str, params: Dict[str, Any]) -> str:
    patched = text
    for key, value in params.items():
        pattern = PARAM_PATTERNS[key]
        if key == "TOM_PASSIVE_EDGE":
            value_str = str(int(round(value)))
        else:
            value_str = f"{float(value):.6f}"
        replacement_count = 0

        def repl(match: re.Match[str]) -> str:
            nonlocal replacement_count
            replacement_count += 1
            return f"{match.group(1)}{value_str}{match.group(3)}"

        patched = pattern.sub(repl, patched, count=1)
        if replacement_count != 1:
            raise ValueError(f"Failed to patch parameter {key}.")
    return patched


def parse_metrics_from_text(text: str) -> TrialMetrics:
    pnl = None
    sharpe = 0.0
    inventory_penalty = 0.0

    for pattern in PNL_PATTERNS:
        m = pattern.search(text)
        if m:
            pnl = float(m.group(1))
            break

    for pattern in SHARPE_PATTERNS:
        m = pattern.search(text)
        if m:
            sharpe = float(m.group(1))
            break

    for pattern in INV_PATTERNS:
        m = pattern.search(text)
        if m:
            inventory_penalty = float(m.group(1))
            break

    if pnl is None:
        # Try whole-text JSON parse as a fallback
        try:
            obj = json.loads(text)
            pnl = float(obj["pnl"])
            sharpe = float(obj.get("sharpe", sharpe))
            inventory_penalty = float(
                obj.get("inventory_penalty", obj.get("inventory_variance", inventory_penalty))
            )
        except Exception:
            raise ValueError(
                "Could not parse pnl from backtest output. "
                "Use metrics_mode json_file/json_stdout/regex_stdout and ensure pnl is exposed."
            )

    return TrialMetrics(
        pnl=pnl,
        sharpe=sharpe,
        inventory_penalty=inventory_penalty,
        raw_output=text,
    )


def load_metrics_from_json_file(path: Path) -> TrialMetrics:
    obj = json.loads(path.read_text())
    return TrialMetrics(
        pnl=float(obj["pnl"]),
        sharpe=float(obj.get("sharpe", 0.0)),
        inventory_penalty=float(obj.get("inventory_penalty", obj.get("inventory_variance", 0.0))),
        raw_output=json.dumps(obj),
    )


def run_one_backtest(
    backtest_cmd_template: str,
    strategy_path: Path,
    metrics_mode: str,
    metrics_path: Optional[Path],
    timeout_sec: int,
) -> TrialMetrics:
    cmd = backtest_cmd_template.format(
        strategy_path=str(strategy_path),
        metrics_path=str(metrics_path) if metrics_path else "",
    )

    result = subprocess.run(
        shlex.split(cmd),
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        check=False,
    )

    combined = (result.stdout or "") + "\n" + (result.stderr or "")

    if metrics_mode == "json_file":
        if metrics_path is None or not metrics_path.exists():
            raise ValueError("metrics_mode=json_file but metrics file was not created.")
        return load_metrics_from_json_file(metrics_path)

    if metrics_mode in {"json_stdout", "regex_stdout"}:
        return parse_metrics_from_text(combined)

    raise ValueError(f"Unsupported metrics_mode: {metrics_mode}")


def compute_score(metrics_list: List[TrialMetrics], sharpe_weight: float, inv_penalty_weight: float) -> float:
    avg_pnl = statistics.mean(m.pnl for m in metrics_list)
    avg_sharpe = statistics.mean(m.sharpe for m in metrics_list)
    avg_inv_penalty = statistics.mean(m.inventory_penalty for m in metrics_list)
    return avg_pnl + sharpe_weight * avg_sharpe - inv_penalty_weight * avg_inv_penalty


def suggest_params(trial: "optuna.trial.Trial") -> Dict[str, Any]:
    return {
        "TOM_FAIR_ALPHA": trial.suggest_float("TOM_FAIR_ALPHA", *PARAM_BOUNDS["TOM_FAIR_ALPHA"]),
        "TOM_PRESSURE_BETA": trial.suggest_float("TOM_PRESSURE_BETA", *PARAM_BOUNDS["TOM_PRESSURE_BETA"]),
        "TOM_TAKE_EDGE": trial.suggest_float("TOM_TAKE_EDGE", *PARAM_BOUNDS["TOM_TAKE_EDGE"]),
        "TOM_PASSIVE_EDGE": trial.suggest_int("TOM_PASSIVE_EDGE", *PARAM_BOUNDS["TOM_PASSIVE_EDGE"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 BO setup for TOMATOES params.")
    parser.add_argument("--base-strategy", type=Path, required=True, help="Path to merged strategy .py file")
    parser.add_argument(
        "--backtest-cmd",
        type=str,
        required=True,
        help="Command template. Use {strategy_path}. Optionally use {metrics_path}.",
    )
    parser.add_argument(
        "--metrics-mode",
        choices=["json_stdout", "json_file", "regex_stdout"],
        default="regex_stdout",
        help="How metrics are provided by your backtester.",
    )
    parser.add_argument("--n-trials", type=int, default=40)
    parser.add_argument("--timeout-sec", type=int, default=120)
    parser.add_argument("--study-name", type=str, default="tomatoes_phase1_bo")
    parser.add_argument("--storage", type=str, default=None, help="Optional Optuna storage, e.g. sqlite:///bo.db")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=1, help="Number of backtest runs per trial")
    parser.add_argument("--sharpe-weight", type=float, default=0.10)
    parser.add_argument("--inv-penalty-weight", type=float, default=0.00)
    parser.add_argument("--output-csv", type=Path, default=Path("bo_phase1_results.csv"))
    args = parser.parse_args()

    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")

    base_text = args.base_strategy.read_text()

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(
        direction="maximize",
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        sampler=sampler,
    )

    csv_exists = args.output_csv.exists()
    csv_file = args.output_csv.open("a", newline="")
    writer = csv.DictWriter(
        csv_file,
        fieldnames=[
            "trial_number",
            "score",
            "avg_pnl",
            "avg_sharpe",
            "avg_inventory_penalty",
            "TOM_FAIR_ALPHA",
            "TOM_PRESSURE_BETA",
            "TOM_TAKE_EDGE",
            "TOM_PASSIVE_EDGE",
        ],
    )
    if not csv_exists:
        writer.writeheader()

    def objective(trial: "optuna.trial.Trial") -> float:
        params = suggest_params(trial)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            strategy_path = tmpdir_path / f"trial_{trial.number}_strategy.py"
            metrics_path = tmpdir_path / f"trial_{trial.number}_metrics.json"

            patched_text = patch_strategy_text(base_text, params)
            strategy_path.write_text(patched_text)

            metrics_list: List[TrialMetrics] = []
            for _ in range(args.repeats):
                metrics = run_one_backtest(
                    backtest_cmd_template=args.backtest_cmd,
                    strategy_path=strategy_path,
                    metrics_mode=args.metrics_mode,
                    metrics_path=metrics_path if args.metrics_mode == "json_file" else None,
                    timeout_sec=args.timeout_sec,
                )
                metrics_list.append(metrics)
                if metrics_path.exists():
                    metrics_path.unlink()

        avg_pnl = statistics.mean(m.pnl for m in metrics_list)
        avg_sharpe = statistics.mean(m.sharpe for m in metrics_list)
        avg_inv_penalty = statistics.mean(m.inventory_penalty for m in metrics_list)

        score = compute_score(
            metrics_list=metrics_list,
            sharpe_weight=args.sharpe_weight,
            inv_penalty_weight=args.inv_penalty_weight,
        )

        trial.set_user_attr("avg_pnl", avg_pnl)
        trial.set_user_attr("avg_sharpe", avg_sharpe)
        trial.set_user_attr("avg_inventory_penalty", avg_inv_penalty)

        writer.writerow({
            "trial_number": trial.number,
            "score": score,
            "avg_pnl": avg_pnl,
            "avg_sharpe": avg_sharpe,
            "avg_inventory_penalty": avg_inv_penalty,
            **params,
        })
        csv_file.flush()

        return score

    study.optimize(objective, n_trials=args.n_trials)

    best = study.best_trial
    summary = {
        "best_score": best.value,
        "best_params": best.params,
        "avg_pnl": best.user_attrs.get("avg_pnl"),
        "avg_sharpe": best.user_attrs.get("avg_sharpe"),
        "avg_inventory_penalty": best.user_attrs.get("avg_inventory_penalty"),
        "trials_completed": len(study.trials),
    }

    print(json.dumps(summary, indent=2))
    csv_file.close()


if __name__ == "__main__":
    main()

