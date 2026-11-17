"""SQLite layer for the Terminal.

One process-wide connection in WAL mode, guarded by a lock for writes.
The uvicorn single-worker model means this is sufficient; if we ever scale
to multiple workers we'd need to move to a connection pool.

Schema is applied idempotently on first import of `get_conn()`.
"""
from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "terminal.sqlite"

_lock = threading.RLock()
_conn: sqlite3.Connection | None = None

_SCHEMA = """
CREATE TABLE IF NOT EXISTS ticker_meta (
    token       TEXT PRIMARY KEY,
    symbol      TEXT,
    exchange    TEXT,
    sector      TEXT,
    industry    TEXT,
    updated_at  INTEGER
);
CREATE INDEX IF NOT EXISTS idx_ticker_meta_sector   ON ticker_meta(sector);
CREATE INDEX IF NOT EXISTS idx_ticker_meta_industry ON ticker_meta(industry);
CREATE INDEX IF NOT EXISTS idx_ticker_meta_exchange ON ticker_meta(exchange);

CREATE TABLE IF NOT EXISTS llm_summaries (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    token         TEXT NOT NULL,
    exchange      TEXT NOT NULL,
    report_type   TEXT NOT NULL,
    report_date   TEXT NOT NULL,
    source_url    TEXT,
    summary_json  TEXT NOT NULL,
    model         TEXT,
    generated_at  INTEGER NOT NULL,
    UNIQUE(token, report_type, report_date)
);
CREATE INDEX IF NOT EXISTS idx_summ_token ON llm_summaries(token);
CREATE INDEX IF NOT EXISTS idx_summ_date  ON llm_summaries(report_date);

CREATE TABLE IF NOT EXISTS relations (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    from_token     TEXT NOT NULL,
    to_name        TEXT NOT NULL,
    to_token       TEXT,
    relation_type  TEXT NOT NULL,
    confidence     REAL NOT NULL DEFAULT 0.5,
    source         TEXT NOT NULL,
    evidence_url   TEXT,
    evidence_text  TEXT,
    extracted_at   INTEGER NOT NULL,
    UNIQUE(from_token, to_name, relation_type, source)
);
CREATE INDEX IF NOT EXISTS idx_rel_from ON relations(from_token);
CREATE INDEX IF NOT EXISTS idx_rel_to   ON relations(to_token);

CREATE TABLE IF NOT EXISTS jobs (
    id          TEXT PRIMARY KEY,
    kind        TEXT NOT NULL,
    token       TEXT,
    exchange    TEXT,
    status      TEXT NOT NULL,
    progress    REAL NOT NULL DEFAULT 0,
    message     TEXT,
    result_ref  TEXT,
    created_at  INTEGER NOT NULL,
    updated_at  INTEGER NOT NULL,
    error       TEXT
);
CREATE INDEX IF NOT EXISTS idx_jobs_token  ON jobs(token);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_kind   ON jobs(kind);

-- PRISM: per-headline FinBERT sentiment cache. article_hash is SHA1 of
-- (title + '|' + url) so identical articles from different fetches dedupe.
CREATE TABLE IF NOT EXISTS news_sentiment (
    article_hash   TEXT PRIMARY KEY,
    token          TEXT,
    title          TEXT,
    url            TEXT,
    published_at   INTEGER,
    score          REAL NOT NULL,      -- signed [-1, 1]: pos - neg
    label          TEXT NOT NULL,      -- positive | negative | neutral
    p_pos          REAL,
    p_neg          REAL,
    p_neu          REAL,
    model_version  TEXT NOT NULL,
    scored_at      INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ns_token ON news_sentiment(token);
CREATE INDEX IF NOT EXISTS idx_ns_pub   ON news_sentiment(published_at);

-- Point-in-time record of the terminal's external signals (analyst consensus,
-- news sentiment, insider activity). Insert-only: one row per (token, UTC day),
-- the FIRST observation of that day, never updated or backfilled, so it can be
-- joined to forecasts later without look-ahead. observed_at is the server clock.
CREATE TABLE IF NOT EXISTS signal_snapshots (
    token              TEXT NOT NULL,
    observed_date      TEXT NOT NULL,      -- YYYY-MM-DD (UTC)
    observed_at        INTEGER NOT NULL,   -- epoch seconds
    symbol             TEXT,
    analyst_key        TEXT,               -- Yahoo recommendationKey
    n_analysts         INTEGER,
    target_mean        REAL,
    ltp                REAL,
    upside_pct         REAL,
    sent_mean          REAL,               -- FinBERT mean over recent headlines
    sent_n             INTEGER,
    insider_buys       INTEGER,
    insider_sells      INTEGER,
    insider_net_value  REAL,
    missing            TEXT,               -- comma list of sources that errored
    PRIMARY KEY (token, observed_date)
);
CREATE INDEX IF NOT EXISTS idx_sigsnap_date ON signal_snapshots(observed_date);

-- Forward test of frozen models (core/forecast/PREREGISTRATION_FORWARD.md).
-- Insert-only: the first score recorded for (model_id, token, asof_date) is kept and
-- never updated, so evaluation can exclude anything recorded after the fact
-- (recorded_at is the server clock, epoch seconds).
CREATE TABLE IF NOT EXISTS forward_predictions (
    model_id     TEXT NOT NULL,
    token        TEXT NOT NULL,
    asof_date    TEXT NOT NULL,      -- YYYY-MM-DD, the session whose close the features end at
    score        REAL NOT NULL,      -- within-day percentile rank of the model's score
    raw_score    REAL,
    regime       TEXT,
    recorded_at  INTEGER NOT NULL,
    PRIMARY KEY (model_id, token, asof_date)
);
CREATE INDEX IF NOT EXISTS idx_fwdpred_date ON forward_predictions(asof_date);

-- PRISM: cached per (token, asof, horizon, model_version) prediction.
-- asof is YYYY-MM-DD; the prediction is for the bar starting that day.
CREATE TABLE IF NOT EXISTS forecasts (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    token             TEXT NOT NULL,
    asof              TEXT NOT NULL,
    horizon           TEXT NOT NULL,     -- '1d' | '5d'
    p_up              REAL,
    expected_return   REAL,
    q10               REAL,
    q50               REAL,
    q90               REAL,
    regime            TEXT,
    attribution_json  TEXT,
    model_version     TEXT NOT NULL,
    created_at        INTEGER NOT NULL,
    UNIQUE(token, asof, horizon, model_version)
);
CREATE INDEX IF NOT EXISTS idx_fc_token  ON forecasts(token);
CREATE INDEX IF NOT EXISTS idx_fc_asof   ON forecasts(asof);

-- PRISM: saved backtest runs. metrics_json + equity_curve_json hold the
-- result payload so the UI can re-render without recomputing.
CREATE TABLE IF NOT EXISTS backtests (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    label              TEXT,
    params_json        TEXT NOT NULL,
    metrics_json       TEXT NOT NULL,
    equity_curve_json  TEXT NOT NULL,
    trades_json        TEXT,
    created_at         INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_bt_created ON backtests(created_at);

-- PRISM: regime state per (date, token). score_json holds the raw inputs
-- (vol, adx-like, vix) so we can re-tune the classifier without re-deriving.
CREATE TABLE IF NOT EXISTS regime_state (
    date        TEXT NOT NULL,
    token       TEXT NOT NULL,
    regime      TEXT NOT NULL,     -- 'trend' | 'meanrev' | 'chop'
    score_json  TEXT,
    PRIMARY KEY (date, token)
);
CREATE INDEX IF NOT EXISTS idx_rs_token ON regime_state(token);

CREATE TABLE IF NOT EXISTS universe_membership (
    universe       TEXT NOT NULL,
    token          TEXT NOT NULL,
    symbol         TEXT NOT NULL,
    effective_from TEXT NOT NULL,
    effective_to   TEXT,
    source         TEXT,
    PRIMARY KEY(universe, token, effective_from)
);
CREATE INDEX IF NOT EXISTS idx_um_dates
    ON universe_membership(universe, effective_from, effective_to);

CREATE TABLE IF NOT EXISTS symbol_history (
    canonical_token TEXT NOT NULL,
    symbol          TEXT NOT NULL,
    exchange        TEXT NOT NULL DEFAULT 'NSE',
    effective_from  TEXT NOT NULL,
    effective_to    TEXT,
    source          TEXT,
    PRIMARY KEY(canonical_token, symbol, effective_from)
);
CREATE INDEX IF NOT EXISTS idx_sh_symbol_dates
    ON symbol_history(symbol, effective_from, effective_to);

CREATE TABLE IF NOT EXISTS corporate_actions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    token         TEXT NOT NULL,
    action_date   TEXT NOT NULL,
    action_type   TEXT NOT NULL,
    price_factor  REAL NOT NULL DEFAULT 1,
    volume_factor REAL NOT NULL DEFAULT 1,
    detail_json   TEXT,
    UNIQUE(token, action_date, action_type)
);
CREATE INDEX IF NOT EXISTS idx_ca_token_date
    ON corporate_actions(token, action_date);

CREATE TABLE IF NOT EXISTS macro_observations (
    series        TEXT NOT NULL,
    observation_date TEXT NOT NULL,
    available_at  TEXT NOT NULL,
    value         REAL NOT NULL,
    source        TEXT,
    PRIMARY KEY(series, observation_date)
);
CREATE INDEX IF NOT EXISTS idx_macro_available
    ON macro_observations(series, available_at);

CREATE TABLE IF NOT EXISTS fundamental_observations (
    token            TEXT NOT NULL,
    metric           TEXT NOT NULL,
    observation_date TEXT NOT NULL,
    available_at     TEXT NOT NULL,
    value            REAL NOT NULL,
    source           TEXT,
    PRIMARY KEY(token, metric, observation_date)
);
CREATE INDEX IF NOT EXISTS idx_fundamental_available
    ON fundamental_observations(token, available_at);

CREATE TABLE IF NOT EXISTS eod_prices (
    token       TEXT NOT NULL,
    trade_date  TEXT NOT NULL,
    open        REAL,
    high        REAL,
    low         REAL,
    close       REAL NOT NULL,
    volume      REAL,
    adjusted    INTEGER NOT NULL DEFAULT 0,
    source      TEXT NOT NULL,
    ingested_at INTEGER NOT NULL,
    PRIMARY KEY(token, trade_date)
);
CREATE INDEX IF NOT EXISTS idx_eod_date ON eod_prices(trade_date);

CREATE TABLE IF NOT EXISTS model_registry (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    model_key      TEXT NOT NULL,
    version        TEXT NOT NULL,
    status         TEXT NOT NULL,
    artifact_path  TEXT NOT NULL,
    metrics_json   TEXT NOT NULL,
    created_at     INTEGER NOT NULL,
    promoted_at    INTEGER,
    UNIQUE(model_key, version)
);
CREATE INDEX IF NOT EXISTS idx_mr_key_status
    ON model_registry(model_key, status);

CREATE TABLE IF NOT EXISTS strategy_registry (
    strategy_id    TEXT NOT NULL,
    version        TEXT NOT NULL,
    name           TEXT NOT NULL,
    status         TEXT NOT NULL,
    universe       TEXT NOT NULL,
    signal_type    TEXT NOT NULL,
    artifact_path  TEXT,
    config_json    TEXT NOT NULL,
    metrics_json   TEXT NOT NULL,
    created_at     INTEGER NOT NULL,
    promoted_at    INTEGER,
    PRIMARY KEY(strategy_id, version)
);
CREATE INDEX IF NOT EXISTS idx_sr_status
    ON strategy_registry(status, universe);

CREATE TABLE IF NOT EXISTS shadow_performance (
    strategy_id   TEXT NOT NULL,
    strategy_version TEXT NOT NULL,
    trade_date    TEXT NOT NULL,
    expected_json TEXT NOT NULL,
    realised_json TEXT,
    status        TEXT NOT NULL DEFAULT 'open',
    created_at    INTEGER NOT NULL,
    updated_at    INTEGER NOT NULL,
    PRIMARY KEY(strategy_id, strategy_version, trade_date)
);

-- Trading runtime. Intent rows are immutable requests; status changes and
-- broker acknowledgements are recorded separately so restarts can reconcile.
CREATE TABLE IF NOT EXISTS trading_settings (
    key         TEXT PRIMARY KEY,
    value_json  TEXT NOT NULL,
    updated_at  INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS order_intents (
    id                TEXT PRIMARY KEY,
    idempotency_key   TEXT NOT NULL UNIQUE,
    token             TEXT NOT NULL,
    symbol            TEXT NOT NULL,
    exchange          TEXT NOT NULL DEFAULT 'NSE',
    side              TEXT NOT NULL,
    quantity          INTEGER NOT NULL,
    order_type        TEXT NOT NULL DEFAULT 'MARKET',
    limit_price       REAL,
    reference_price   REAL,
    mode              TEXT NOT NULL,
    status            TEXT NOT NULL,
    signal_json       TEXT,
    risk_json         TEXT,
    reason            TEXT,
    created_at        INTEGER NOT NULL,
    updated_at        INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_oi_status ON order_intents(status);
CREATE INDEX IF NOT EXISTS idx_oi_created ON order_intents(created_at);

CREATE TABLE IF NOT EXISTS execution_reports (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    intent_id       TEXT NOT NULL,
    broker_order_id TEXT,
    status          TEXT NOT NULL,
    filled_qty      INTEGER NOT NULL DEFAULT 0,
    avg_price       REAL,
    fees            REAL NOT NULL DEFAULT 0,
    raw_json        TEXT,
    created_at      INTEGER NOT NULL,
    FOREIGN KEY(intent_id) REFERENCES order_intents(id)
);
CREATE INDEX IF NOT EXISTS idx_er_intent ON execution_reports(intent_id);

CREATE TABLE IF NOT EXISTS trading_positions (
    token         TEXT PRIMARY KEY,
    symbol        TEXT NOT NULL,
    exchange      TEXT NOT NULL,
    quantity      INTEGER NOT NULL,
    avg_price     REAL NOT NULL,
    last_price    REAL,
    realised_pnl  REAL NOT NULL DEFAULT 0,
    updated_at    INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS trading_audit (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type  TEXT NOT NULL,
    entity_id   TEXT,
    detail_json TEXT NOT NULL,
    created_at  INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ta_created ON trading_audit(created_at);

CREATE TABLE IF NOT EXISTS server_watchlists (
    name        TEXT NOT NULL DEFAULT 'default',
    token       TEXT NOT NULL,
    symbol      TEXT NOT NULL,
    exchange    TEXT NOT NULL,
    added_at    INTEGER NOT NULL,
    PRIMARY KEY(name, token)
);
"""


def get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is not None:
        return _conn
    with _lock:
        if _conn is not None:
            return _conn
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(
            str(DB_PATH),
            check_same_thread=False,
            isolation_level=None,  # autocommit; explicit txns via BEGIN
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.executescript(_SCHEMA)
        _conn = conn
        return conn


def lock() -> threading.RLock:
    """Shared write lock. Readers do not need to take it thanks to WAL mode."""
    return _lock
