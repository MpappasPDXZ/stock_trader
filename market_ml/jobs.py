"""Job tracker + stage runner for the market-ml sidecar.

Reuses ``market_ml.pipeline._STAGE_SOURCES`` / ``_STAGE_ORDER`` so the sidecar never
duplicates the feature stack — it execs the exact notebook-extracted stages. The only
addition vs ``run_ticker`` is that ``run_stages`` returns the exec globals ``g`` so
``models.py`` can pull out the fitted ``live_xgb`` / ``live_lasso`` / featureset for saving.

Jobs run one at a time in a single worker thread (XGBoost is CPU-bound and stages use
module-level ``sys.stdout`` redirection for log capture, so serializing keeps logs clean).
"""
from __future__ import annotations

import io
import re
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any

from market_ml.pipeline import (
    RunConfig,
    _STAGE_ORDER,
    _STAGE_SOURCES,
    _apply_overrides,
    _patch_model_n_jobs,
    run_ticker,
)


# --------------------------------------------------------------------------- #
# Exec globals seed (mirrors run_ticker in extract.py — same imports/helpers)
# --------------------------------------------------------------------------- #
def _seed_globals(g: dict[str, Any]) -> dict[str, Any]:
    import json as _json
    import warnings as _warnings
    from datetime import datetime as _dt, time as _time
    from pathlib import Path as _Path
    from zoneinfo import ZoneInfo as _ZI

    import matplotlib.pyplot as _plt
    import numpy as _np
    import pandas as _pd
    from sklearn.linear_model import Lasso as _Lasso, LassoCV as _LassoCV
    from sklearn.metrics import (
        accuracy_score as _acc,
        classification_report as _cr,
        confusion_matrix as _cm,
        mean_absolute_error as _mae,
        mean_squared_error as _mse,
        r2_score as _r2,
        roc_auc_score as _roc,
        roc_curve as _rocf,
    )
    from sklearn.model_selection import TimeSeriesSplit as _tss
    from sklearn.pipeline import Pipeline as _Pipe
    from sklearn.preprocessing import StandardScaler as _SS
    from xgboost import XGBClassifier as _XGB

    g.update({
        "__name__": "market_ml.jobs",
        "display": _display_shim,
        "np": _np, "pd": _pd, "plt": _plt, "Path": _Path,
        "warnings": _warnings, "json": _json,
        "datetime": _dt, "time": _time, "ZoneInfo": _ZI,
        "Pipeline": _Pipe, "StandardScaler": _SS,
        "Lasso": _Lasso, "LassoCV": _LassoCV, "TimeSeriesSplit": _tss,
        "XGBClassifier": _XGB,
        "accuracy_score": _acc, "classification_report": _cr,
        "confusion_matrix": _cm, "mean_absolute_error": _mae,
        "mean_squared_error": _mse, "r2_score": _r2,
        "roc_auc_score": _roc, "roc_curve": _rocf,
    })

    import matplotlib.dates as mdates
    from matplotlib.patches import Rectangle, Patch
    from matplotlib.lines import Line2D
    from scipy import stats
    from scipy.stats import spearmanr

    g["mdates"] = mdates
    g["Rectangle"] = Rectangle
    g["Patch"] = Patch
    g["Line2D"] = Line2D
    g["stats"] = stats
    g["spearmanr"] = spearmanr

    # Defined fresh here (same as run_ticker) so plot_01/plot_03 work even when
    # feature_health is skipped.
    import numpy as np

    def annotate_prev_close_right(ax, close: float, *, linewidth: float = 1.15) -> None:
        try:
            c = float(close)
        except (TypeError, ValueError):
            return
        if not np.isfinite(c):
            return
        _ = linewidth
        ax.annotate(
            f"← ${c:.2f}",
            xy=(1.0, c),
            xycoords=("axes fraction", "data"),
            xytext=(6, 0),
            textcoords="offset points",
            ha="left", va="center", color="black",
            fontsize=9, fontweight="bold",
            clip_on=False, annotation_clip=False, zorder=8,
        )

    g["annotate_prev_close_right"] = annotate_prev_close_right
    return g


def _display_shim(obj: Any = None, *args, **kwargs) -> None:
    if obj is None:
        return
    try:
        if hasattr(obj, "to_string"):
            print(obj.to_string())
        else:
            print(obj)
    except Exception:
        print(repr(obj))


# --------------------------------------------------------------------------- #
# Stage runner (returns globals) — mirrors run_ticker's exec loop without the
# summary extraction, so callers can reach fitted objects / featureset in g.
# --------------------------------------------------------------------------- #
def run_stages(
    cfg: RunConfig,
    *,
    up_to: str | None = None,
    skip: tuple[str, ...] = (),
    quiet: bool = False,
    _globals: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Exec canonical stages into a globals dict and return (g, summary).

    ``up_to`` runs stages through that stage (inclusive). ``skip`` drops stages.
    ``_globals`` — if provided, resume into an existing globals dict (skips
    stages already executed, identified by keys they set).  Starts fresh otherwise.
    """
    if _globals is not None:
        g = _globals
    else:
        g: dict[str, Any] = {"__builtins__": __builtins__}
        _seed_globals(g)

    stages = list(_STAGE_ORDER)
    if up_to is not None:
        if up_to not in stages:
            raise ValueError(f"unknown stage {up_to!r}; have {_STAGE_ORDER}")
        stages = stages[: stages.index(up_to) + 1]
    for s in skip:
        stages = [s2 for s2 in stages if s2 != s]

    summary: dict[str, Any] = {"ok": False, "error": None, "stages_done": []}
    try:
        for stage in stages:
            src = _patch_model_n_jobs(_STAGE_SOURCES[stage], cfg.n_jobs_model)
            if stage == "setup":
                src = re.sub(
                    r'^TICKER\s*=\s*["\'][^"\']*["\']',
                    f'TICKER = {cfg.ticker.upper().strip()!r}',
                    src, count=1, flags=re.M,
                )
                exec(compile(src, f"<stage:{stage}>", "exec"), g, g)
                _apply_overrides(cfg, g)
            else:
                exec(compile(src, f"<stage:{stage}>", "exec"), g, g)
            if stage == "feature_registry" and cfg.active_features is not None:
                af = set(cfg.active_features)
                g["FEATURE_SPECS"] = [s for s in g["FEATURE_SPECS"] if s["name"] in af]
                g["FEATURE_COLS"] = [s["name"] for s in g["FEATURE_SPECS"]]
                g["FEATURE_MAX_SESSIONS"] = max(s["sessions_before_start"] for s in g["FEATURE_SPECS"])
                if not quiet:
                    print(f"[{g['TICKER']}] active_features override: {g['FEATURE_COLS']}")
            summary["stages_done"].append(stage)
            if not quiet:
                print(f"[{g.get('TICKER', cfg.ticker)}] stage ok: {stage}")
        summary["ok"] = True
    except Exception as e:
        summary["ok"] = False
        summary["error"] = f"{type(e).__name__}: {e}"
        summary["traceback"] = traceback.format_exc()
    return g, summary


# --------------------------------------------------------------------------- #
# Job tracker — one at a time, stdout captured per job
# --------------------------------------------------------------------------- #
@dataclass
class Job:
    id: str
    kind: str
    args: dict[str, Any]
    status: str = "queued"          # queued | running | done | error
    logs: list[str] = field(default_factory=list)
    result: dict[str, Any] | None = None
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    _done: threading.Event = field(default_factory=threading.Event, repr=False)

    def to_public(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "args": self.args,
            "status": self.status,
            "n_logs": len(self.logs),
            "result": self.result,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
        }


class _JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._queue: list[str] = []
        self._worker = threading.Thread(target=self._loop, daemon=True)
        self._worker.start()

    # -- public API --
    def submit(self, job_id: str, kind: str, fn, args: dict[str, Any]) -> Job:
        job = Job(id=job_id, kind=kind, args=args)
        with self._lock:
            self._jobs[job_id] = job
            self._queue.append(job_id)
        # fn is called in the worker thread with (job, args)
        self._fns[job_id] = (fn, args)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def all(self) -> list[Job]:
        with self._lock:
            return list(self._jobs.values())

    # -- worker loop --
    def _loop(self) -> None:
        self._fns: dict[str, Any] = {}
        while True:
            with self._lock:
                if not self._queue:
                    pass
                job_id = self._queue.pop(0) if self._queue else None
            if job_id is None:
                time.sleep(0.05)
                continue
            fn, args = self._fns.pop(job_id)
            self._run_job(job_id, fn, args)

    def _run_job(self, job_id: str, fn, args: dict[str, Any]) -> None:
        job = self.get(job_id)
        if job is None:
            return
        job.status = "running"
        buf = _LineStream(job.logs)
        old_stdout = sys.stdout
        sys.stdout = buf  # type: ignore[assignment]
        try:
            result = fn(job, args)
            job.result = result
            job.status = "done" if (result and result.get("ok", True)) else "error"
        except Exception as e:
            job.status = "error"
            job.result = {"ok": False, "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}
        finally:
            sys.stdout = old_stdout
            job.finished_at = time.time()
            job._done.set()


JOBS = _JobStore()


class _LineStream(io.TextIOBase):
    """Append-only stdout sink that splits into log lines for a Job."""
    def __init__(self, sink: list[str]) -> None:
        self._sink = sink
        self._buf = ""

    def write(self, s: str) -> int:
        if not s:
            return 0
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._sink.append(line)
        return len(s)

    def flush(self) -> None:  # noqa: D401
        if self._buf:
            self._sink.append(self._buf)
            self._buf = ""
