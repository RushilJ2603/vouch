"""Section 11 -- point-in-time integrity, enforced structurally.

The cheapest section to implement and the most expensive to get wrong.
Calibration measured with leaked information is not calibration, and a leaked
curve looks BETTER than an honest one, which is why the failure is silent.
"""
import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "vouch"

# proxy/app.py legitimately stamps the present moment. Nothing else may.
WALL_CLOCK_ALLOWED = {"proxy/app.py"}

WALL_CLOCK = re.compile(r"time\.time\(\)|datetime\.now\(|utcnow")


def _offenders() -> list[str]:
    hits = []
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC).as_posix()
        if rel in WALL_CLOCK_ALLOWED:
            continue
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            if WALL_CLOCK.search(line):
                hits.append(f"{rel}:{n}: {line.strip()}")
    return hits


def test_no_wall_clock_outside_the_proxy():
    """Every time-relative value is computed against the decision's own
    timestamp, never against wall clock. `age = decision.ts - row.created_at`,
    never `age = time.time() - row.created_at`."""
    offenders = _offenders()
    assert offenders == [], (
        "wall-clock call outside proxy/app.py -- this silently biases "
        "calibration and historical replays:\n  " + "\n  ".join(offenders)
    )


def test_the_check_can_actually_fail(tmp_path):
    """A CI check nobody has seen fail is a CI check nobody knows works."""
    assert WALL_CLOCK.search("now = time.time()")
    assert WALL_CLOCK.search("datetime.now(timezone.utc)")
    assert WALL_CLOCK.search("datetime.utcnow()")
    assert not WALL_CLOCK.search("age = decision.ts - row.created_at")
