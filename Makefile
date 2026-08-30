.PHONY: demo corpus test lint seed worker up down clean

PY := PYTHONPATH=src python3

# ── The demos must run with NO API key set. That is a hard constraint: a
# judge clones the repo and runs this. Everything replays from the committed
# corpus; nothing here reaches the network.
demo:
	@for d in 01_calibration 02_autonomy 03_latency 04_cost; do \
	  if [ -f scripts/demo_$$d.py ]; then \
	    echo "--- demo_$$d ---"; $(PY) scripts/demo_$$d.py || exit 1; \
	  else \
	    echo "--- demo_$$d: not built yet (Week 2-4) ---"; \
	  fi; \
	done

# ★ The only Make target that spends money. Batched, offline, run off-peak.
corpus:
	@test -n "$$DEEPSEEK_API_KEY" || { echo "DEEPSEEK_API_KEY not set (see .env.example)"; exit 1; }
	@test -n "$$ZAI_API_KEY"      || { echo "ZAI_API_KEY not set (see .env.example)"; exit 1; }
	$(PY) scripts/generate_corpus.py
	$(PY) scripts/judge_corpus.py

test: lint
	$(PY) -m pytest

# The §11 point-in-time gate is part of lint, not just of the test suite, so it
# runs even when someone skips the tests. Wall clock outside proxy/app.py
# silently biases calibration and makes historical replays non-reproducible.
lint:
	@echo "point-in-time integrity (§11):"
	@! grep -rnE "time\.time\(\)|datetime\.now\(|utcnow" src/vouch/ \
	   --exclude-dir=__pycache__ | grep -v "^src/vouch/proxy/app.py" \
	   || { echo "  FAIL: wall-clock call outside proxy/app.py"; exit 1; }
	@echo "  clean"
	@command -v ruff >/dev/null || { echo "  FAIL: ruff not installed (it is pinned in requirements.lock)"; exit 1; }
	ruff check src/ scripts/ tests/ dashboard/

# The worker takes its reference time from the caller; see 11 and
# worker/__main__.py. Never reads the clock itself.
worker:
	$(PY) -m vouch.worker --as-of $$(date +%s)

seed:
	$(PY) scripts/seed_db.py

up:
	docker compose up -d --build

down:
	docker compose down

clean:
	rm -f data/orders.sqlite data/orders.sqlite-wal data/orders.sqlite-shm
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
