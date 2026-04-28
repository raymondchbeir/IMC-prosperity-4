from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from app.config import (
    RUST_BACKTESTER_ARTIFACT_MODE,
    RUST_BACKTESTER_BIN,
    RUST_BACKTESTER_PRODUCTS_MODE,
)
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

    run_start = time.perf_counter()
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
            "--artifact-mode", RUST_BACKTESTER_ARTIFACT_MODE,
            "--products", RUST_BACKTESTER_PRODUCTS_MODE,
        ]

        if request.selected_day is not None and request.preset in {"selected_dashboard_round_day", "manual"}:
            targets = _tokens_to_targets(request)
            if len(targets) == 1:
                cmd.extend(["--day", str(targets[0].day_num)])

        env = os.environ.copy()
        repo_root = Path(__file__).resolve().parents[2]
        env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")

        rust_start = time.perf_counter()
        proc = subprocess.run(cmd, cwd=str(repo_root), env=env, text=True, capture_output=True)
        rust_seconds = time.perf_counter() - rust_start
        if proc.returncode != 0:
            raise RuntimeError(
                "rust_backtester failed.\n\nSTDOUT:\n"
                + proc.stdout[-4000:]
                + "\n\nSTDERR:\n"
                + proc.stderr[-4000:]
            )

        log_path = _find_best_log(output_root)
        parse_start = time.perf_counter()
        portal_payload = parse_portal_files([log_path])
        parse_seconds = time.perf_counter() - parse_start

        payload_start = time.perf_counter()
        targets = _tokens_to_targets(request)
        per_run_rows = _build_per_run_rows(portal_payload.summary, targets)

        summary = dict(portal_payload.summary)
        summary.setdefault("targets", [t.label for t in targets])
        summary.setdefault("num_runs", len(targets) or 1)
        summary["extra_volume_pct"] = float(request.extra_volume_pct or 0.0)
        summary["backend"] = "rust_backtester"
        summary["rust_log_path"] = str(log_path)
        summary["artifact_mode"] = RUST_BACKTESTER_ARTIFACT_MODE
        summary["products_mode"] = RUST_BACKTESTER_PRODUCTS_MODE
        summary["timings"] = {
            "rust_seconds": round(rust_seconds, 3),
            "parse_seconds": round(parse_seconds, 3),
            "payload_seconds": round(time.perf_counter() - payload_start, 3),
            "total_seconds": round(time.perf_counter() - run_start, 3),
        }
        print("RUST_BACKTEST_TIMINGS", summary["timings"])

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

    dataset_alias = _dataset_alias_from_request(request)
    resolved = _resolve_local_dataset_alias(dataset_alias)
    return resolved if resolved is not None else dataset_alias


def _dataset_alias_from_request(request: BacktestRequest) -> str:
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


def _resolve_local_dataset_alias(dataset_alias: str) -> Path | None:
    """Prefer concrete dataset directories so aliases like round3 work from the dashboard repo.

    rust_backtester resolves short aliases relative to the process cwd. The dashboard runs it from the
    IMC dashboard repo, while the bundled Rust datasets usually live in tools/prosperity_rust_backtester.
    """
    if not isinstance(dataset_alias, str):
        return None

    repo_root = Path(__file__).resolve().parents[2]
    alias_map = {
        "latest": None,
        "tutorial": "tutorial",
        "tut": "tutorial",
        "tutorial-round": "tutorial",
        "tut-round": "tutorial",
        "round1": "round1",
        "r1": "round1",
        "round2": "round2",
        "r2": "round2",
        "round3": "round3",
        "r3": "round3",
        "round4": "round4",
        "r4": "round4",
        "round5": "round5",
        "r5": "round5",
        "round6": "round6",
        "r6": "round6",
        "round7": "round7",
        "r7": "round7",
        "round8": "round8",
        "r8": "round8",
    }
    key = dataset_alias.lower()
    folder = alias_map.get(key)
    candidate_roots = [
        Path(os.getenv("RUST_BACKTESTER_DATASETS", "")) if os.getenv("RUST_BACKTESTER_DATASETS") else None,
        repo_root / "tools" / "prosperity_rust_backtester" / "datasets",
        repo_root / "datasets",
        repo_root / "data",
    ]

    if folder is None and key == "latest":
        for root in candidate_roots:
            if root is None or not root.exists():
                continue
            rounds = [p for p in root.iterdir() if p.is_dir() and (p / "").exists()]
            populated = [p for p in rounds if any(p.glob("prices_*.csv")) or any(p.glob("*.log")) or any(p.glob("*.json"))]
            if populated:
                return sorted(populated, key=lambda p: p.name)[-1]
        return None

    if folder is None:
        return None

    for root in candidate_roots:
        if root is None:
            continue
        candidate = root / folder
        if candidate.exists():
            return candidate
    return None


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
