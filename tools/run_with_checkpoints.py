"""Replay the paper's 25-fold walk-forward and keep every trained model.

This is a thin wrapper around `tools/regenerate_mixture_ic_series.py`. It does not
change the computation: it wraps `ranking.train_ranker` so that each model the
replay trains (one pooled model and up to three regime experts per fold) is also
written to disk, then calls the original `main()`, which compares every statistic
against `data/regime_mixture_result.json` and fails if any differs by > 1e-6.

Outputs (default under `outputs/replay/`):
  checkpoints/fold_XX_<pooled|trend|meanrev|chop>.txt.gz   LightGBM model text
  checkpoints/manifest.json                                 per-model metadata + sha256
  logs/replay_<UTC timestamp>.log                           full console log
  reproduction_check.json, ic_by_date.csv, ic_by_fold.csv   from the original script

Load a checkpoint:
  import gzip, lightgbm as lgb
  booster = lgb.Booster(model_str=gzip.open(path, "rt").read())

Usage:
  python tools/run_with_checkpoints.py [--out outputs/replay]
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

import lightgbm  # noqa: E402
import numpy  # noqa: E402
import pandas  # noqa: E402
import scipy  # noqa: E402

import regenerate_mixture_ic_series as replay_script  # noqa: E402
from core.forecast import ranking  # noqa: E402


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
        return len(s)

    def flush(self):
        for st in self.streams:
            st.flush()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="outputs/replay")
    args, _ = parser.parse_known_args()
    out_dir = ROOT / args.out
    ckpt_dir = out_dir / "checkpoints"
    log_dir = out_dir / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = log_dir / f"replay_{stamp}.log"
    log_fh = log_path.open("w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, log_fh)
    sys.stderr = _Tee(sys.__stderr__, log_fh)

    print(f"run started (UTC): {stamp}")
    print(f"python {platform.python_version()} on {platform.platform()}")
    print(f"numpy {numpy.__version__} | pandas {pandas.__version__} | "
          f"scipy {scipy.__version__} | lightgbm {lightgbm.__version__}")
    manifest_src = ROOT / "data" / "research" / "nifty_dataset_manifest.json"
    if manifest_src.exists():
        dev = json.loads(manifest_src.read_text(encoding="utf-8")).get("development", {})
        print(f"dataset manifest development partition: {json.dumps(dev)}")

    original = ranking.train_ranker
    records: list[dict] = []
    state = {"fold": 0}

    def saving_train_ranker(train, config, *, feature_cols=None):
        artifact = original(train, config, feature_cols=feature_cols)
        regimes = train["regime"].dropna().unique() if "regime" in train.columns else []
        label = str(regimes[0]) if len(regimes) == 1 else "pooled"
        if label == "pooled":
            state["fold"] += 1
        name = f"fold_{state['fold']:02d}_{label}.txt.gz"
        path = ckpt_dir / name
        with gzip.open(path, "wt", encoding="utf-8", compresslevel=9) as fh:
            fh.write(artifact.model.model_to_string())
        dates = train["date"]
        records.append({
            "file": name,
            "fold": state["fold"],
            "model": label,
            "train_rows": int(len(train)),
            "train_date_min": str(dates.min())[:10],
            "train_date_max": str(dates.max())[:10],
            "n_features": len(artifact.feature_cols),
            "feature_cols": list(artifact.feature_cols),
            "ranking_config": artifact.config,
            "sha256": _sha256(path),
        })
        return artifact

    ranking.train_ranker = saving_train_ranker
    sys.argv = [sys.argv[0], "--out", str(Path(args.out) / "."), ]
    try:
        code = replay_script.main()
    finally:
        ranking.train_ranker = original
        (ckpt_dir / "manifest.json").write_text(
            json.dumps({"n_models": len(records), "models": records}, indent=1) + "\n",
            encoding="utf-8",
        )
        print(f"saved {len(records)} checkpoints + manifest to {ckpt_dir}")
        print(f"log: {log_path}")
        log_fh.flush()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
