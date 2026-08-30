# One image, three services (§6.1). The proxy, the worker and the dashboard
# differ only in their command, so they share a layer cache.
FROM python:3.12-slim

WORKDIR /app

# Dependencies first so source edits do not invalidate the install layer.
COPY requirements.lock ./
RUN pip install --no-cache-dir -r requirements.lock

COPY src/ ./src/
COPY config/ ./config/
COPY scripts/ ./scripts/
COPY dashboard/ ./dashboard/
COPY data/ /data/
COPY artifacts/ /artifacts/
COPY reports/ ./reports/

ENV PYTHONPATH=/app/src
ENV VOUCH_CONFIG_DIR=/app/config

CMD ["python", "-c", "print('specify a command: see docker-compose.yml')"]
