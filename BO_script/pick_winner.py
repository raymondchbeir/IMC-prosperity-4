"""Sort completed Optuna trials by val PnL (held-out 100k) and show the top 5
with their full parameter sets.  Also computes median-of-top-5 as a robust
"clustered" winner.

Usage:
    PYTHONPATH=. python BO_script/pick_winner.py
"""
import json
import statistics

import optuna

STUDY_NAME = "bo_round4_final"
STORAGE    = "sqlite:///optuna_study.db"
BASELINE_VAL = 55347.5    # patched trader on the 100k held-out (no overrides)
TOP_N      = 5


def main():
    study = optuna.load_study(study_name=STUDY_NAME, storage=STORAGE)
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not done:
        print("No completed trials yet."); return

    ranked = sorted(done,
                    key=lambda t: t.user_attrs.get("val_pnl", -1e18),
                    reverse=True)
    top = ranked[:TOP_N]

    print(f"baseline val PnL: {BASELINE_VAL:,.0f}")
    print(f"completed trials: {len(done)}")
    print(f"showing top {TOP_N} ranked by val PnL\n")
    print("=" * 78)

    # --- Per-trial detail ---------------------------------------------------
    for i, t in enumerate(top):
        val   = t.user_attrs.get("val_pnl", float("nan"))
        train = t.value if t.value is not None else float("nan")
        train_pnls = t.user_attrs.get("train_pnls", [])
        delta = val - BASELINE_VAL

        print(f"\n#{i+1}   trial_number={t.number}")
        print(f"     val PnL   : {val:>12,.0f}   ({delta:+,.0f} vs baseline)")
        print(f"     train mean: {train:>12,.0f}")
        if train_pnls:
            tp_str = ", ".join(f"{p:,.0f}" for p in train_pnls)
            print(f"     train pnls: [{tp_str}]")
        print(f"     params:")
        for k, v in sorted(t.params.items()):
            if isinstance(v, float):
                print(f"       {k:<28} = {v:.4f}")
            else:
                print(f"       {k:<28} = {v}")

    print("\n" + "=" * 78)

    # --- Side-by-side parameter comparison ----------------------------------
    print(f"\nside-by-side parameter view (top {TOP_N}):\n")
    keys = sorted(top[0].params.keys())
    header = f"{'param':<28}" + "".join(f"{'#'+str(i+1):>14}" for i in range(len(top)))
    print(header)
    print("-" * len(header))
    for k in keys:
        row = f"{k:<28}"
        for t in top:
            v = t.params[k]
            if isinstance(v, float):
                row += f"{v:>14.4f}"
            else:
                row += f"{v:>14}"
        print(row)

    # --- Median-of-top-N clustered params -----------------------------------
    print("\n" + "=" * 78)
    clustered = {}
    for k in top[0].params:
        vals = [t.params[k] for t in top]
        if isinstance(vals[0], int):
            clustered[k] = int(round(statistics.median(vals)))
        else:
            clustered[k] = round(statistics.median(vals), 4)

    best_val = top[0].user_attrs.get("val_pnl", -1e18)
    print(f"\ntop-1 val: {best_val:,.0f}   ({best_val - BASELINE_VAL:+,.0f} vs baseline)")
    print(f"\nclustered params (median of top {TOP_N}):")
    print(json.dumps(clustered, indent=2))

    out_path = "BO_script/clustered_params.json"
    json.dump(clustered, open(out_path, "w"), indent=2)
    print(f"\n→ saved {out_path}")

    # --- Decision hint ------------------------------------------------------
    delta_pct = 100 * (best_val - BASELINE_VAL) / BASELINE_VAL if BASELINE_VAL else 0
    print(f"\ntop-1 vs baseline: {delta_pct:+.1f}%")
    if delta_pct >= 10:
        print("→ STRONG: apply clustered params, expect real improvement.")
    elif delta_pct >= 5:
        print("→ MODERATE: apply clustered params but verify on training days first.")
    elif delta_pct >= 0:
        print("→ WEAK: marginal gain.  Consider submitting baseline instead.")
    else:
        print("→ NEGATIVE: BO did not improve val.  Submit baseline.")


if __name__ == "__main__":
    main()
