"""Per-ticker sector/industry metadata — persisted in SQLite, queried by
`instruments.search()` so users can type "banking" or "IT services" and
get matching stocks.

Enriched lazily (on fundamentals fetch) and in bulk by `core.enricher`.
"""
from __future__ import annotations

import time

from .db import get_conn, lock


def upsert(
    token: str,
    symbol: str,
    exchange: str,
    sector: str | None,
    industry: str | None,
) -> None:
    """Write a meta row. NULL sector/industry is allowed — we still record the
    attempt so the enricher doesn't retry the same ticker every pass.

    Existing non-null sector/industry is preserved if the new row is NULL
    (COALESCE), so an opportunistic call from the fundamentals path can't
    wipe out good data from a bad yfinance response.
    """
    conn = get_conn()
    now = int(time.time())
    with lock():
        conn.execute(
            """
            INSERT INTO ticker_meta (token, symbol, exchange, sector, industry, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(token) DO UPDATE SET
                symbol=excluded.symbol,
                exchange=excluded.exchange,
                sector=COALESCE(excluded.sector, ticker_meta.sector),
                industry=COALESCE(excluded.industry, ticker_meta.industry),
                updated_at=excluded.updated_at
            """,
            (token, symbol, exchange, sector, industry, now),
        )


def get(token: str) -> dict | None:
    conn = get_conn()
    row = conn.execute("SELECT * FROM ticker_meta WHERE token = ?", (token,)).fetchone()
    return dict(row) if row else None


def load_all() -> dict[str, dict]:
    """Entire table as {token: row_dict}. Small enough to keep in memory
    (<10k rows typical)."""
    conn = get_conn()
    rows = conn.execute(
        "SELECT token, symbol, exchange, sector, industry FROM ticker_meta"
    ).fetchall()
    return {r["token"]: dict(r) for r in rows}


def search_tokens_by_category(query: str, exchange: str, limit: int = 500) -> list[str]:
    """Tokens whose sector OR industry contains `query` (case-insensitive).
    Used as one of several ranked buckets in instruments.search()."""
    q = f"%{query.lower()}%"
    conn = get_conn()
    rows = conn.execute(
        """
        SELECT token FROM ticker_meta
        WHERE exchange = ?
          AND (LOWER(sector) LIKE ? OR LOWER(industry) LIKE ?)
        LIMIT ?
        """,
        (exchange, q, q, limit),
    ).fetchall()
    return [r["token"] for r in rows]


def coverage_stats() -> dict:
    """Quick diagnostic: how far along is enrichment?"""
    conn = get_conn()
    total = conn.execute("SELECT COUNT(*) AS c FROM ticker_meta").fetchone()["c"]
    with_sector = conn.execute(
        "SELECT COUNT(*) AS c FROM ticker_meta WHERE sector IS NOT NULL"
    ).fetchone()["c"]
    return {"rows": total, "with_sector": with_sector}
