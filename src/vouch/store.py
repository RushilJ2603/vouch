import re
import sqlite3

# ─────────────────────────────────────────────────────────────────────────
# Schema
# ─────────────────────────────────────────────────────────────────────────

SCHEMA_SQL = """PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- ─────────────────────────────────────────────────────────────────────────
-- The ledger. One row per (agent, action, band, config). Written by the
-- worker only; the proxy holds a read-only copy in memory (§12.2).
-- ─────────────────────────────────────────────────────────────────────────
CREATE TABLE trust_row (
    agent_id           TEXT    NOT NULL,
    action             TEXT    NOT NULL,
    band               TEXT    NOT NULL,          -- '0-2k' | '2k-10k' | '10k-50k' | '50k+'
    config_fingerprint TEXT    NOT NULL,          -- §9.4; part of the identity, not a column
    n_total            REAL    NOT NULL DEFAULT 0,-- REAL: weighted by evidence and age (§9.6)
    n_clean            REAL    NOT NULL DEFAULT 0,
    n_reversed         REAL    NOT NULL DEFAULT 0,
    n_own_raw          INTEGER NOT NULL DEFAULT 0,-- unweighted, un-borrowed; gates borrowing (§9.3)
    p_lo               REAL    NOT NULL DEFAULT 0,-- Wilson lower bound
    budget             REAL    NOT NULL DEFAULT 0,
    ceiling            REAL    NOT NULL,          -- derived (§5.2); stored so decisions are auditable
    state              TEXT    NOT NULL DEFAULT 'supervised',  -- supervised | autonomous
    tripped_at         REAL,                      -- circuit breaker (§9.5)
    clean_since_trip   INTEGER NOT NULL DEFAULT 0,
    psi_short          REAL,
    psi_long           REAL,
    psi_drift          REAL,
    b_short_ref        TEXT,                      -- JSON: binned reference distribution
    b_long_ref         TEXT,                      -- JSON: immutable until Level 3 (§19)
    updated_at         REAL    NOT NULL,
    PRIMARY KEY (agent_id, action, band, config_fingerprint)
);

-- ─────────────────────────────────────────────────────────────────────────
-- Every decision. This is the calibration dataset, the latency ledger and
-- the audit trail at once. Appended by the proxy; never updated by it.
-- ─────────────────────────────────────────────────────────────────────────
CREATE TABLE decision (
    id                  TEXT PRIMARY KEY,         -- ULID, sortable by time
    ts                  REAL    NOT NULL,         -- THE reference time for all of §11
    agent_id            TEXT    NOT NULL,
    action              TEXT    NOT NULL,
    band                TEXT    NOT NULL,
    fingerprint         TEXT    NOT NULL,
    calibrator_version  TEXT    NOT NULL,         -- excludes mixed-version windows (§12.6)
    capability_tier     TEXT    NOT NULL,         -- full | topk | none (§7.2)
    features            TEXT    NOT NULL,         -- JSON, the full vector + _unavailable flags
    claims              TEXT,                     -- JSON, the agent's typed claims
    p_wrong             REAL    NOT NULL,
    challenger_p_wrong  REAL,                     -- §18.1 Path B shadow; NULL otherwise
    tier_reached        INTEGER NOT NULL,         -- 0 | 1 | 2
    exposure            REAL    NOT NULL,
    budget              REAL    NOT NULL,         -- as it WAS, not as it is now (§11)
    expected_loss       REAL    NOT NULL,
    verdict             TEXT    NOT NULL,         -- act | check_harder | escalate | block
    region              TEXT    NOT NULL,         -- act | close_call | escalate — stamped, not derived
    rung                INTEGER NOT NULL,         -- 0..6 (§10.3)
    outcome             TEXT,                     -- NULL until the horizon closes (§13 Step 1)
    outcome_source      TEXT,                     -- record_verification | human_override | reversal | user_rework | offline_judge
    outcome_ts          REAL,
    horizon_ends        REAL    NOT NULL,
    latency_ms          REAL    NOT NULL,         -- proxy total
    upstream_ms         REAL    NOT NULL,         -- first byte to last byte; added = latency − upstream
    degraded_mode       INTEGER NOT NULL DEFAULT 0,
    degraded_reason     TEXT,                     -- §16 reason codes
    shadow              INTEGER NOT NULL DEFAULT 0,
    forced_review       INTEGER NOT NULL DEFAULT 0 -- the ~1-in-300 hold-out (§7.7)
);

-- ─────────────────────────────────────────────────────────────────────────
-- One row per nightly drift check per trust row. The audit trail of every
-- monitoring verdict, including the ones that took no action.
-- ─────────────────────────────────────────────────────────────────────────
CREATE TABLE drift_check_log (
    run_ts         REAL    NOT NULL,
    agent_id       TEXT    NOT NULL,
    action         TEXT    NOT NULL,
    band           TEXT    NOT NULL,
    psi_short      REAL,
    psi_long       REAL,
    psi_drift      REAL,
    ece_act        REAL,
    ece_close_call REAL,
    ece_escalate   REAL,
    n_observations INTEGER NOT NULL,              -- < 100 → insufficient_data, no alarm (§19)
    verdict        TEXT    NOT NULL,              -- ok | insufficient_data | re_anchor | L1 | L2 | L3
    action_taken   TEXT    NOT NULL,
    detail         TEXT,                          -- JSON
    PRIMARY KEY (run_ts, agent_id, action, band)
);

-- ─────────────────────────────────────────────────────────────────────────
-- Which calibrator is production, and the reload handshake (§12.6).
-- ─────────────────────────────────────────────────────────────────────────
CREATE TABLE calibrator_registry (
    version              TEXT PRIMARY KEY,        -- 'cal-v3'
    created_at           REAL    NOT NULL,
    promoted_at          REAL,
    ready_at             REAL,                    -- NULL until artifact integrity confirmed
    state                TEXT    NOT NULL,        -- candidate | shadow | production | retired
    promotion_path       TEXT,                    -- A | B
    sklearn_version      TEXT    NOT NULL,        -- hard-checked at load (§2.8)
    capability_tier      TEXT    NOT NULL,
    ece_global           REAL,
    ece_close_call       REAL,
    ece_delta_close_call REAL,                    -- vs the outgoing version; drives migration (§18.2)
    rows_carried         INTEGER,
    rows_halved          INTEGER,
    rows_reset           INTEGER,
    artifact_sha256      TEXT    NOT NULL,
    notes                TEXT
);

-- ─────────────────────────────────────────────────────────────────────────
-- Post-promotion evolving baselines. APPEND ONLY — never updated in place,
-- so the re-anchor history stays auditable (§9.7).
-- ─────────────────────────────────────────────────────────────────────────
CREATE TABLE model_baseline (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    calibrator_version TEXT    NOT NULL,
    effective_ts       REAL    NOT NULL,
    source             TEXT    NOT NULL,          -- reanchor | live_lock
    agent_id           TEXT,
    action             TEXT,
    band               TEXT,
    b_short            TEXT,                      -- JSON
    ece_baselines      TEXT,                      -- JSON, per region
    FOREIGN KEY (calibrator_version) REFERENCES calibrator_registry(version)
);

-- ─────────────────────────────────────────────────────────────────────────
-- Raw outcome reports. Enqueued by the proxy; resolved by worker Step 1
-- against the horizon. The proxy never writes decision.outcome directly.
-- ─────────────────────────────────────────────────────────────────────────
CREATE TABLE outcome_event (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    decision_id  TEXT    NOT NULL,
    reported_ts  REAL    NOT NULL,
    outcome      TEXT    NOT NULL,                -- clean | wrong
    source       TEXT    NOT NULL,
    weight       REAL    NOT NULL,                -- from EVIDENCE_WEIGHT (§9.6)
    processed_at REAL,                            -- NULL until worker Step 1 consumes it
    FOREIGN KEY (decision_id) REFERENCES decision(id)
);

-- ─────────────────────────────────────────────────────────────────────────
-- Indexes
-- ─────────────────────────────────────────────────────────────────────────
CREATE INDEX idx_decision_pending   ON decision(horizon_ends) WHERE outcome IS NULL;
CREATE INDEX idx_decision_row       ON decision(agent_id, action, band, ts);
CREATE INDEX idx_decision_forced    ON decision(forced_review) WHERE forced_review = 1;
CREATE INDEX idx_decision_region    ON decision(region, p_wrong);
CREATE INDEX idx_decision_cal       ON decision(calibrator_version, ts);
CREATE INDEX idx_outcome_unprocessed ON outcome_event(processed_at) WHERE processed_at IS NULL;
CREATE INDEX idx_drift_run          ON drift_check_log(run_ts);
"""


# ─────────────────────────────────────────────────────────────────────────
# Connection and Setup
# ─────────────────────────────────────────────────────────────────────────


def connect(
    path: str, *, read_only: bool = False, check_same_thread: bool = True
) -> sqlite3.Connection:
    """Connect to SQLite database and set required pragmas."""
    uri_path = f"file:{path}?mode=ro" if read_only and path != ":memory:" else path
    conn = sqlite3.connect(
        uri_path, uri=(read_only and path != ":memory:"), check_same_thread=check_same_thread
    )
    conn.row_factory = sqlite3.Row
    if not read_only:
        conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


# SCHEMA_SQL is Appendix C verbatim, so the IF NOT EXISTS needed for
# idempotency is applied here rather than by editing the spec. Anchored to the
# start of a line and guarded against an existing IF NOT EXISTS, so it cannot
# double-substitute or rewrite the words inside a `--` comment.
_IDEMPOTENT = re.compile(r"^CREATE (TABLE|INDEX)(?! IF NOT EXISTS)", re.MULTILINE)


def init_schema(conn: sqlite3.Connection) -> None:
    """Initialize database schema idempotently."""
    conn.executescript(_IDEMPOTENT.sub(r"CREATE \1 IF NOT EXISTS", SCHEMA_SQL))


# ─────────────────────────────────────────────────────────────────────────
# Data Accessors
# ─────────────────────────────────────────────────────────────────────────


def upsert_trust_row(conn: sqlite3.Connection, row: dict) -> None:
    """Insert or update a single trust row."""
    keys = list(row.keys())
    placeholders = ", ".join(["?"] * len(keys))
    cols = ", ".join(keys)
    set_clause = ", ".join(
        [
            f"{k}=EXCLUDED.{k}"
            for k in keys
            if k not in ["agent_id", "action", "band", "config_fingerprint"]
        ]
    )

    sql = f"""
    INSERT INTO trust_row ({cols}) 
    VALUES ({placeholders})
    ON CONFLICT(agent_id, action, band, config_fingerprint) 
    DO UPDATE SET {set_clause}
    """
    conn.execute(sql, [row[k] for k in keys])


def get_trust_row(
    conn: sqlite3.Connection, agent_id: str, action: str, band: str, config_fingerprint: str
) -> dict | None:
    """Retrieve a single trust row by its identity."""
    sql = """
    SELECT * FROM trust_row 
    WHERE agent_id = ? AND action = ? AND band = ? AND config_fingerprint = ?
    """
    cursor = conn.execute(sql, (agent_id, action, band, config_fingerprint))
    row = cursor.fetchone()
    return dict(row) if row else None


def load_all_trust_rows(conn: sqlite3.Connection) -> list[dict]:
    """Retrieve all trust rows."""
    cursor = conn.execute("SELECT * FROM trust_row")
    return [dict(row) for row in cursor.fetchall()]


def append_decision(conn: sqlite3.Connection, decision: dict) -> None:
    """Insert a new decision record."""
    keys = list(decision.keys())
    placeholders = ", ".join(["?"] * len(keys))
    cols = ", ".join(keys)
    sql = f"INSERT INTO decision ({cols}) VALUES ({placeholders})"
    conn.execute(sql, [decision[k] for k in keys])
