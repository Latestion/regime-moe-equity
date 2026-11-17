"""FinBERT news-sentiment service.

We score every headline + (optional) summary through ProsusAI/finbert and
cache the result in `news_sentiment` keyed by SHA1(title|url). The cache
makes backfills and re-renders cheap: a watchlist re-render hits the DB,
not the model.

The transformer model is loaded lazily on first use — startup of the
Terminal stays fast for users who never open the forecast tab. Model
inference runs on CPU (the AMD Radeon path on Windows isn't worth the
complexity for an 110M-param model that hits ~10 ms/headline on a 14600K).
"""
from __future__ import annotations

import asyncio
import datetime as _dt
import hashlib
import logging
import threading
import time
from typing import Any, Iterable

from ..db import get_conn, lock

log = logging.getLogger("forecast.sentiment")

MODEL_NAME    = "ProsusAI/finbert"
MODEL_VERSION = "finbert-v1"   # bump when you change the model or labels
_LABELS       = ("positive", "negative", "neutral")  # FinBERT's canonical order

_BATCH_SIZE   = 16
_MAX_LEN      = 256

# Lazy singletons. Loading transformers + torch is ~0.5-1s and pulls the
# model into RAM (~440 MB), so we only do it when score_*() is first called.
_init_lock = threading.Lock()
_tokenizer = None
_model     = None
_torch     = None


def _load() -> None:
    """Idempotent lazy initialisation. Safe to call from multiple threads."""
    global _tokenizer, _model, _torch
    if _model is not None:
        return
    with _init_lock:
        if _model is not None:
            return
        log.info("loading FinBERT (%s) — first call only", MODEL_NAME)
        import torch  # noqa: WPS433  (local import is intentional)
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        mdl = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        mdl.eval()
        # Validate label ordering — the published config maps id->label this way
        # but we should fail loudly if it ever changes upstream.
        cfg = getattr(mdl, "config", None)
        id2label = getattr(cfg, "id2label", None) if cfg else None
        if id2label:
            ordered = [id2label[i].lower() for i in sorted(id2label)]
            if tuple(ordered) != _LABELS:
                log.warning(
                    "FinBERT label order changed upstream: %s (expected %s) — "
                    "scores will be reordered to match expected layout",
                    ordered, _LABELS,
                )
                # Build a permutation so logits[:, perm] gives [pos, neg, neu].
                perm = [ordered.index(l) for l in _LABELS]
                mdl._label_perm = perm  # type: ignore[attr-defined]
        _tokenizer = tok
        _model     = mdl
        _torch     = torch


def article_hash(title: str | None, url: str | None) -> str:
    key = f"{title or ''}|{url or ''}".encode("utf-8", errors="replace")
    return hashlib.sha1(key).hexdigest()


def _to_unix_ts(value: Any) -> int | None:
    """Best-effort conversion of yfinance/news ``published`` to a Unix int.

    yfinance hands us either an int/float (legacy schema) or an ISO 8601
    string (current schema after `core.news._to_iso`). Both must end up as
    seconds-since-epoch so the SQL date filters in `aggregates_for_token`
    can compare them.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return None  # avoid the int(bool) trap
    if isinstance(value, (int, float)):
        v = int(value)
        # yfinance sometimes hands us milliseconds; detect by magnitude.
        return v // 1000 if v > 10_000_000_000 else v
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        # Normalise trailing Z → +00:00 for fromisoformat (pre-3.11 compat).
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        try:
            dt = _dt.datetime.fromisoformat(s)
        except ValueError:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=_dt.timezone.utc)
        return int(dt.timestamp())
    return None


def _split_text(s: str | None) -> str:
    return (s or "").strip().replace("\n", " ")


def score_batch(texts: list[str]) -> list[dict]:
    """Run FinBERT on a list of texts; return one dict per input with
    keys ``score`` (signed in [-1, 1]: p_pos - p_neg), ``label``, and
    the three class probabilities ``p_pos`` / ``p_neg`` / ``p_neu``.

    Empty strings are returned as zero-confidence neutrals without touching
    the model — saves time on the common case where summary is missing.
    """
    _load()
    assert _model is not None and _tokenizer is not None and _torch is not None

    out: list[dict] = []
    work_idx: list[int] = []
    work_txt: list[str] = []
    for i, t in enumerate(texts):
        tt = _split_text(t)
        if not tt:
            out.append({
                "score": 0.0, "label": "neutral",
                "p_pos": 0.0, "p_neg": 0.0, "p_neu": 1.0,
            })
        else:
            out.append({})  # placeholder
            work_idx.append(i)
            work_txt.append(tt[:2000])  # truncate before tokenising

    if not work_txt:
        return out

    perm = getattr(_model, "_label_perm", None)
    torch = _torch
    with torch.no_grad():
        for start in range(0, len(work_txt), _BATCH_SIZE):
            batch = work_txt[start:start + _BATCH_SIZE]
            enc = _tokenizer(
                batch, padding=True, truncation=True,
                max_length=_MAX_LEN, return_tensors="pt",
            )
            logits = _model(**enc).logits  # (B, 3)
            if perm is not None:
                logits = logits[:, perm]
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            for j, p in enumerate(probs):
                p_pos, p_neg, p_neu = float(p[0]), float(p[1]), float(p[2])
                label = _LABELS[int(p.argmax())]
                out[work_idx[start + j]] = {
                    "score": p_pos - p_neg,
                    "label": label,
                    "p_pos": p_pos,
                    "p_neg": p_neg,
                    "p_neu": p_neu,
                }
    return out


# ---------------------------------------------------------------------------
# Cache layer
# ---------------------------------------------------------------------------


def _fetch_cached(hashes: list[str]) -> dict[str, dict]:
    if not hashes:
        return {}
    # SQLite has a 999-parameter limit; chunk just in case.
    out: dict[str, dict] = {}
    conn = get_conn()
    for i in range(0, len(hashes), 800):
        chunk = hashes[i:i + 800]
        q = ",".join("?" for _ in chunk)
        rows = conn.execute(
            f"""SELECT article_hash, score, label, p_pos, p_neg, p_neu, model_version
                FROM news_sentiment
                WHERE article_hash IN ({q}) AND model_version = ?""",
            (*chunk, MODEL_VERSION),
        ).fetchall()
        for r in rows:
            out[r["article_hash"]] = dict(r)
    return out


def _upsert(rows: list[tuple]) -> None:
    if not rows:
        return
    with lock():
        get_conn().executemany(
            """INSERT INTO news_sentiment
                  (article_hash, token, title, url, published_at,
                   score, label, p_pos, p_neg, p_neu, model_version, scored_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(article_hash) DO UPDATE SET
                  score=excluded.score, label=excluded.label,
                  p_pos=excluded.p_pos, p_neg=excluded.p_neg, p_neu=excluded.p_neu,
                  model_version=excluded.model_version, scored_at=excluded.scored_at""",
            rows,
        )


def score_news_items(
    items: Iterable[dict],
    *,
    token: str | None = None,
) -> list[dict]:
    """Score a list of normalised news items (shape produced by `core.news`).

    Returns the input list augmented with a ``sentiment`` key per item.
    Cache-first: items already in `news_sentiment` are looked up; only the
    misses go through the model. Caches new scores back to the DB.
    """
    items_list = list(items)
    if not items_list:
        return []

    # Build hashes + indices.
    hashes = [article_hash(it.get("title"), it.get("url")) for it in items_list]
    cached = _fetch_cached(hashes)

    miss_idx: list[int] = []
    miss_texts: list[str] = []
    for i, it in enumerate(items_list):
        h = hashes[i]
        if h in cached:
            it["sentiment"] = {
                "score": cached[h]["score"],
                "label": cached[h]["label"],
                "p_pos": cached[h]["p_pos"],
                "p_neg": cached[h]["p_neg"],
                "p_neu": cached[h]["p_neu"],
                "cached": True,
            }
        else:
            miss_idx.append(i)
            title = it.get("title") or ""
            summary = it.get("summary") or ""
            miss_texts.append((title + ". " + summary).strip(". "))

    if miss_texts:
        scored = score_batch(miss_texts)
        now = int(time.time())
        rows: list[tuple] = []
        for j, s in enumerate(scored):
            i = miss_idx[j]
            it = items_list[i]
            it["sentiment"] = {**s, "cached": False}
            pub_ts = _to_unix_ts(it.get("published"))
            rows.append((
                hashes[i], token, it.get("title"), it.get("url"), pub_ts,
                s["score"], s["label"], s["p_pos"], s["p_neg"], s["p_neu"],
                MODEL_VERSION, now,
            ))
        _upsert(rows)

    return items_list


async def score_news_items_async(
    items: Iterable[dict],
    *,
    token: str | None = None,
) -> list[dict]:
    """Async wrapper — runs the (sync, CPU-bound) scorer in a thread."""
    return await asyncio.to_thread(score_news_items, list(items), token=token)


# ---------------------------------------------------------------------------
# Convenience aggregations for the feature pipeline
# ---------------------------------------------------------------------------


def aggregates_for_token(
    token: str,
    *,
    asof_ts: int | None = None,
    windows: tuple[int, ...] = (1, 7, 30),
) -> dict[str, float | int | None]:
    """Return rolling sentiment aggregates for ``token`` ending at ``asof_ts``.

    For each window in days we return ``mean_<n>d``, ``dispersion_<n>d``
    (std of scores), and ``count_<n>d``. Used by `core/forecast/features.py`.
    """
    asof = int(asof_ts) if asof_ts is not None else int(time.time())
    out: dict[str, float | int | None] = {}
    conn = get_conn()
    for w in windows:
        lo = asof - w * 86400
        row = conn.execute(
            """SELECT AVG(score) AS mean, COUNT(*) AS n,
                      AVG(score*score) AS msq
               FROM news_sentiment
               WHERE token = ? AND published_at IS NOT NULL
                 AND published_at BETWEEN ? AND ?""",
            (token, lo, asof),
        ).fetchone()
        n = (row["n"] if row else 0) or 0
        mean = (row["mean"] if row and row["mean"] is not None else None)
        msq  = (row["msq"]  if row and row["msq"]  is not None else None)
        disp: float | None = None
        if mean is not None and msq is not None and n >= 2:
            var = max(0.0, msq - mean * mean)
            disp = var ** 0.5
        out[f"mean_{w}d"]       = mean
        out[f"dispersion_{w}d"] = disp
        out[f"count_{w}d"]      = int(n)
    return out
