"""Store schema and idempotency tests."""

from vouch import store

APPENDIX_C_TABLES = {
    "trust_row", "decision", "drift_check_log",
    "calibrator_registry", "model_baseline", "outcome_event",
}

def test_all_six_tables_are_created():
    conn = store.connect(":memory:")
    store.init_schema(conn)
    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert APPENDIX_C_TABLES <= names


def test_init_schema_is_idempotent():
    conn = store.connect(":memory:")
    store.init_schema(conn)
    store.init_schema(conn)          # must not raise
    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert APPENDIX_C_TABLES <= names


def test_idempotency_substitution_cannot_double_apply():
    """The rewrite is anchored and guarded. Applying it to already-rewritten
    SQL must be a no-op, and it must not touch the words inside a comment."""
    once = store._IDEMPOTENT.sub(r"CREATE \1 IF NOT EXISTS", store.SCHEMA_SQL)
    twice = store._IDEMPOTENT.sub(r"CREATE \1 IF NOT EXISTS", once)
    assert once == twice
    assert "IF NOT EXISTS IF NOT EXISTS" not in twice
    comment = "-- CREATE TABLE something, described in prose\nCREATE TABLE real_one (a INT);"
    assert store._IDEMPOTENT.sub(r"CREATE \1 IF NOT EXISTS", comment).startswith(
        "-- CREATE TABLE something")


def test_trust_row_primary_key_is_the_full_identity():
    """(agent, action, band, config_fingerprint). The fingerprint is part of
    the identity, not a column -- that is what makes trust expiry structural."""
    conn = store.connect(":memory:")
    store.init_schema(conn)
    pk = [r["name"] for r in conn.execute("PRAGMA table_info(trust_row)") if r["pk"]]
    assert pk == ["agent_id", "action", "band", "config_fingerprint"]
