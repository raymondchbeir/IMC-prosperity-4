"""Optuna Bayesian-optimization study, wired to the Rust backtester.

Each trial:
  1. Optuna samples a parameter set.
  2. We write a temp strategy .py file with Trader.<NAME> = <value> overrides.
  3. The Rust backtester runs it on each train day.
  4. We aggregate per-day PnLs into a risk-adjusted score (mean - λ·std).
  5. MedianPruner kills clearly-bad trials after the first couple of days.

Walk-forward: optimize on `--train-days`, then evaluate the locked-in best
params on `--val-days` (held out from optimization).  The val score is your
honest out-of-sample number.

Usage:
    python optuna_study.py --train-days 4-1 4-2 4-3 \\
                           --val-days   4-4 4-5 \\
                           --trials 200 --n-jobs 4

Day tokens use `round-day` (e.g. `4-1`).  Negative days as `4--1`.

Outputs:
    optuna_study.db     SQLite study (resumable; re-run to add more trials)
    best_params.json    Best params on the training fold
    walkforward.json    Train + held-out PnL summary
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

# Adapter that runs one (round, day) through the Rust backtester.
from rust_backtest_wrapper import run_one_day


# ---------------------------------------------------------------------------
# 1. Parameter space.
# ---------------------------------------------------------------------------
# Picked by: (a) parameters with v## annotations showing past tuning paid off,
# (b) parameters that govern signal/quoting decisions where small changes
# meaningfully shift PnL.  Keep the space TIGHT — broad bounds waste budget
# in obviously-bad regions.  Re-center the bounds around your current values
# and widen only if BO consistently pushes to an edge.
#
# Spec format: (kind, *args).  kind in {"float", "log_float", "int", "cat"}.
PARAM_SPACE: Dict[str, tuple] = {
    # ---- VELVET signal core ----
    "Z_THRESHOLD":          ("float",   2.5,   3.3),
    "PRICE_OVERRIDE_DIST":  ("float",  18.0,  30.0),
    "ADVERSE_STOP_TICKS":   ("float",   8.0,  22.0),
    "COOLDOWN_TS":          ("int",    3000, 12000),
    "REGIME_TIMEOUT_TS":    ("int",   20000,100000),

    # ---- VELVET execution & trim ----
    "TAKE_FRACTION":        ("float",   0.05,  0.60),
    "MAX_PASSIVE_PER_TICK": ("int",      50,   120),
    "TRIM_HARD_FAIR_DIST":  ("float",   6.0,  16.0),
    "TRIM_KEEP_FRACTION":   ("float",   0.40,  0.85),

    # ---- Voucher overlay ----
    "VOUCHER_BASE_TARGET":   ("int",    60,   140),
    "VOUCHER_DIST_PER_TICK": ("float",   3.0,   8.0),
    "VOUCHER_MARK_FRACTION": ("float",   0.50,  1.00),
    "VOUCHER_TAKE_FRACTION": ("float",   0.30,  1.00),
    "PEAK_EXTRA_MIN_DIST":   ("float",  15.0,  30.0),
    "PEAK_EXTRA_MIN_Z":      ("float",   2.50,  3.50),

    # ---- MM ----
    "VELVET_MM_QUOTE_SIZE":   ("int",     3,    12),
    "VELVET_MM_MIN_SPREAD":   ("int",     2,     8),
    "VOUCHER_MM_QUOTE_SIZE":  ("int",     2,     8),
    "VOUCHER_MM_CONVERGED_TOL":("int",    2,    12),

    # ---- HG (Hydrogel) — biggest historical wins ----
    "HG_TILT_K":          ("float",  0.5,   3.0),
    "HG_TILT_CAP":        ("int",     40,   140),
    "HG_TAKE_THRESHOLD":  ("int",     20,    45),
    "HG_TAKE_SIZE":       ("int",      4,    16),
    "HG_TAKE_CAP":        ("int",     80,   240),
    "HG_MM_QUOTE_SIZE":   ("int",      6,    20),
    "HG_MM_POS_SOFT":     ("int",     50,   140),
    "HG_MM_POS_HARD":     ("int",    120,   220),
}


def suggest(trial: optuna.Trial, name: str, spec: tuple) -> Any:
    kind, *args = spec
    if kind == "float":     return trial.suggest_float(name, *args)
    if kind == "log_float": return trial.suggest_float(name, *args, log=True)
    if kind == "int":       return trial.suggest_int(name, *args)
    if kind == "cat":       return trial.suggest_categorical(name, args[0])
    raise ValueError(f"unknown kind: {kind!r}")


# ---------------------------------------------------------------------------
# 2. Day-token parser.  CLI accepts "4-1" (round-day), "4--1" for negative days.
# ---------------------------------------------------------------------------
def parse_day_token(tok: str) -> Tuple[int, int]:
    if "--" in tok:
        r, d = tok.split("--", 1)
        return int(r), -int(d)
    if "-" in tok:
        r, d = tok.split("-", 1)
        return int(r), int(d)
    raise ValueError(f"day token must be 'R-D' (got {tok!r})")


# ---------------------------------------------------------------------------
# 3. Param sanity-fixups (cheap interaction guards before sending to backtest).
# ---------------------------------------------------------------------------
def _fixup(params: Dict[str, Any]) -> Dict[str, Any]:
    p = dict(params)
    # HG soft cap must be below hard cap.
    if "HG_MM_POS_SOFT" in p and "HG_MM_POS_HARD" in p:
        if p["HG_MM_POS_SOFT"] >= p["HG_MM_POS_HARD"]:
            p["HG_MM_POS_SOFT"] = p["HG_MM_POS_HARD"] - 10
    return p


# ---------------------------------------------------------------------------
# 4. Risk-adjusted objective with median pruning across days.
# ---------------------------------------------------------------------------
RISK_AVERSION = 0.5   # mean(pnl) - lambda * std(pnl).  Increase to penalise variance.


def evaluate(params: Dict[str, Any],
             days: Sequence[Tuple[int, int]],
             trial: optuna.Trial | None = None) -> Tuple[float, List[float]]:
    """Run Rust backtest on each (round, day); return (score, per-day pnls)."""
    pnls: List[float] = []
    for i, (rnd, dy) in enumerate(days):
        pnl = run_one_day(params, rnd, dy)
        pnls.append(pnl)
        if trial is not None:
            trial.report(statistics.mean(pnls), step=i)
            if trial.should_prune():
                raise optuna.TrialPruned()
    mean = statistics.mean(pnls)
    std  = statistics.pstdev(pnls) if len(pnls) > 1 else 0.0
    return mean - RISK_AVERSION * std, pnls


def make_objective(train_days: Sequence[Tuple[int, int]]):
    def objective(trial: optuna.Trial) -> float:
        params = {name: suggest(trial, name, spec)
                  for name, spec in PARAM_SPACE.items()}
        params = _fixup(params)
        score, pnls = evaluate(params, train_days, trial=trial)
        trial.set_user_attr("pnls", pnls)
        return score
    return objective


# ---------------------------------------------------------------------------
# 5. Walk-forward main.
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-days", nargs="+", required=True,
                    help="Round-day tokens to optimise on, e.g. 4-1 4-2 4-3.")
    ap.add_argument("--val-days",   nargs="+", required=True,
                    help="Held-out tokens.  Best params are scored here AFTER tuning.")
    ap.add_argument("--trials",     type=int, default=200)
    ap.add_argument("--n-jobs",     type=int, default=1,
                    help="Parallel trial workers.  Each rust_backtester call is "
                         "isolated in its own TemporaryDirectory, so >1 is safe.")
    ap.add_argument("--storage",    default="sqlite:///optuna_study.db")
    ap.add_argument("--name",       default="trader_bo")
    ap.add_argument("--seed",       type=int, default=0)
    args = ap.parse_args()

    train_days = [parse_day_token(t) for t in args.train_days]
    val_days   = [parse_day_token(t) for t in args.val_days]

    sampler = TPESampler(n_startup_trials=20, multivariate=True, group=True,
                         seed=args.seed)
    pruner  = MedianPruner(n_startup_trials=10, n_warmup_steps=2)
    study   = optuna.create_study(
        direction="maximize",
        sampler=sampler, pruner=pruner,
        storage=args.storage, study_name=args.name,
        load_if_exists=True,
    )
    study.optimize(make_objective(train_days),
                   n_trials=args.trials, n_jobs=args.n_jobs,
                   show_progress_bar=True)

    best = _fixup(study.best_params)
    val_score, val_pnls = evaluate(best, val_days)

    Path("best_params.json").write_text(json.dumps(best, indent=2))
    Path("walkforward.json").write_text(json.dumps({
        "train_days":         [f"{r}-{d}" for r, d in train_days],
        "val_days":           [f"{r}-{d}" for r, d in val_days],
        "train_best_score":   study.best_value,
        "train_best_pnls":    study.best_trial.user_attrs.get("pnls"),
        "val_score":          val_score,
        "val_pnls":           val_pnls,
        "params":             best,
        "n_trials":           len(study.trials),
        "n_pruned":           sum(1 for t in study.trials
                                  if t.state == optuna.trial.TrialState.PRUNED),
    }, indent=2))

    print(f"\nbest train score: {study.best_value:,.0f}")
    print(f"out-of-sample   : {val_score:,.0f}   pnls={val_pnls}")
    print("→ saved best_params.json, walkforward.json")


if __name__ == "__main__":
    main()
