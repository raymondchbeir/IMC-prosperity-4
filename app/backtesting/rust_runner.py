from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from app.config import RUST_BACKTESTER_BIN
from app.models.schemas import BacktestPayload, BacktestRequest, BacktestTarget
from app.portal_logs.parser import parse_portal_files

DATA_FILE_RE_SUFFIXES = (".csv", ".json", ".log")


def run_rust_backtests(
    strategy_path: Path,
    request: BacktestRequest,
    custom_data_root: Path | None = None,
) -> BacktestPayload:
    """Run GeyzsoN's rust_backtester CLI and convert its submission log into the dashboard payload.

    The Rust backtester is a command-line tool, not an importable Python API. This adapter saves/copies
    the needed inputs into a temporary run folder, executes rust_backtester, then parses the generated
    submission.log/combined.log using app.portal_logs.parser.
    """
    if shutil.which(RUST_BACKTESTER_BIN) is None:
        raise RuntimeError(
            f"Could not find `{RUST_BACKTESTER_BIN}` on PATH. Run `cargo install rust_backtester --locked` and make sure ~/.cargo/bin is on PATH."
        )

    with TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        output_root = tmpdir / "rust_runs"
        output_root.mkdir(parents=True, exist_ok=True)

        local_strategy = _prepare_strategy(strategy_path, tmpdir)
        dataset_path = _resolve_dataset_path(request, custom_data_root, tmpdir)

        if request.extra_volume_pct and float(request.extra_volume_pct) > 0:
            dataset_path = _build_extra_volume_dataset(dataset_path, tmpdir, float(request.extra_volume_pct))

        cmd = [
            RUST_BACKTESTER_BIN,
            "--trader", str(local_strategy),
            "--dataset", str(dataset_path),
            "--output-root", str(output_root),
            "--persist",
            "--artifact-mode", "full",
            "--trade-match-mode", request.match_trades or "all",
        ]

        if request.selected_day is not None and request.preset in {"selected_dashboard_round_day", "manual"}:
            targets = _tokens_to_targets(request)
            if len(targets) == 1:
                cmd.extend(["--day", str(targets[0].day_num)])

        env = os.environ.copy()
        repo_root = Path(__file__).resolve().parents[2]
        env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")

        proc = subprocess.run(cmd, cwd=str(repo_root), env=env, text=True, capture_output=True)
        if proc.returncode != 0:
            raise RuntimeError(
                "rust_backtester failed.\n\nSTDOUT:\n"
                + proc.stdout[-4000:]
                + "\n\nSTDERR:\n"
                + proc.stderr[-4000:]
            )

        log_path = _find_best_log(output_root)
        portal_payload = parse_portal_files([log_path])
        targets = _tokens_to_targets(request)
        per_run_rows = _build_per_run_rows(portal_payload.summary, targets)

        summary = dict(portal_payload.summary)
        summary.setdefault("targets", [t.label for t in targets])
        summary.setdefault("num_runs", len(targets) or 1)
        summary["extra_volume_pct"] = float(request.extra_volume_pct or 0.0)
        summary["backend"] = "rust_backtester"
        summary["rust_log_path"] = str(log_path)

        return BacktestPayload(
            targets=targets,
            activity_rows=portal_payload.activity_rows,
            submission_trade_rows=portal_payload.submission_trade_rows,
            realized_trade_rows=portal_payload.realized_trade_rows,
            sandbox_rows=portal_payload.sandbox_rows,
            per_run_rows=per_run_rows,
            per_product_rows=portal_payload.per_product_rows,
            summary=summary,
        )


def _prepare_strategy(strategy_path: Path, tmpdir: Path) -> Path:
    strategy_dir = tmpdir / "strategy"
    strategy_dir.mkdir(parents=True, exist_ok=True)
    local_strategy = strategy_dir / strategy_path.name
    local_strategy.write_bytes(Path(strategy_path).read_bytes())

    repo_root = Path(__file__).resolve().parents[2]
    datamodel = repo_root / "datamodel.py"
    if datamodel.exists():
        (strategy_dir / "datamodel.py").write_bytes(datamodel.read_bytes())
    return local_strategy


def _resolve_dataset_path(request: BacktestRequest, custom_data_root: Path | None, tmpdir: Path) -> Path | str:
    if custom_data_root is not None:
        return custom_data_root

    expr = (request.targets_text or "").strip()
    round_num = request.selected_round
    if request.preset == "selected_dashboard_round" and round_num is not None:
        return f"round{int(round_num)}"
    if request.preset == "selected_dashboard_round_day" and round_num is not None:
        return f"round{int(round_num)}"
    if expr:
        first = expr.replace(",", " ").split()[0]
        try:
            round_part = first.split("-", 1)[0]
            return f"round{int(round_part)}"
        except Exception:
            pass
    return "latest"


def _tokens_to_targets(request: BacktestRequest) -> list[BacktestTarget]:
    expr = (request.targets_text or "").strip()
    if request.preset == "selected_dashboard_round" and request.selected_round is not None:
        return [BacktestTarget(round_num=int(request.selected_round), day_num=0)]
    if request.preset == "selected_dashboard_round_day" and request.selected_round is not None and request.selected_day is not None:
        return [BacktestTarget(round_num=int(request.selected_round), day_num=int(request.selected_day))]
    out: list[BacktestTarget] = []
    for token in expr.replace(",", " ").split():
        if "--" in token:
            r, d = token.split("--", 1)
            out.append(BacktestTarget(round_num=int(r), day_num=-int(d)))
        elif "-" in token:
            r, d = token.split("-", 1)
            out.append(BacktestTarget(round_num=int(r), day_num=int(d)))
        elif token.strip().isdigit():
            out.append(BacktestTarget(round_num=int(token), day_num=0))
    return out or [BacktestTarget(round_num=0, day_num=0)]


def _build_extra_volume_dataset(dataset_path: Path | str, tmpdir: Path, extra_volume_pct: float) -> Path | str:
    if not isinstance(dataset_path, Path):
        # Built-in aliases live inside the Rust package. Do not mutate them here.
        # The dashboard still records the setting; uploaded custom data gets actual volume scaling.
        return dataset_path

    source = dataset_path
    target = tmpdir / "extra_volume_dataset"
    shutil.copytree(source, target, dirs_exist_ok=True)
    factor = 1.0 + extra_volume_pct
    for csv_path in target.rglob("prices_round_*_day_*.csv"):
        _scale_prices_file(csv_path, factor)
    return target


def _scale_prices_file(csv_path: Path, factor: float) -> None:
    df = pd.read_csv(csv_path, sep=";", engine="python")
    for col in ["bid_volume_1", "bid_volume_2", "bid_volume_3", "ask_volume_1", "ask_volume_2", "ask_volume_3"]:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce")
            df[col] = vals.where(vals.isna(), (vals * factor).round().astype("Int64"))
    df.to_csv(csv_path, sep=";", index=False)


def _find_best_log(output_root: Path) -> Path:
    candidates = list(output_root.rglob("submission.log"))
    if not candidates:
        candidates = list(output_root.rglob("combined.log"))
    if not candidates:
        raise RuntimeError(f"rust_backtester completed but no submission.log/combined.log was found under {output_root}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _build_per_run_rows(summary: dict, targets: list[BacktestTarget]) -> list[dict]:
    return [
        {
            "round": t.round_num,
            "run_day": t.day_num,
            "run_label": t.label,
            "final_pnl": summary.get("total_pnl"),
            "submission_trades": summary.get("submission_trade_count", 0),
            "realized_trade_count": summary.get("realized_trade_count", 0),
            "winning_trade_count": summary.get("winning_trade_count", 0),
            "losing_trade_count": summary.get("losing_trade_count", 0),
        }
        for t in (targets or [BacktestTarget(round_num=0, day_num=0)])
    ]
