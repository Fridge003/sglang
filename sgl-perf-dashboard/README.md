# sgl-perf-dashboard

Web dashboard for tracking sglang GB200 nightly benchmark regressions.

Source: [DLR-5301 design doc](https://jirasw.nvidia.com/browse/DLR-5301).

## Layout

- `api/` — FastAPI backend + MinIO ingester + SQLite store
- `web/` — Next.js 15 frontend (Tailwind + shadcn/ui)
- `docker-compose.yaml` — production deployment (on the ci-logs Brev box)
- `docker-compose.dev.yaml` — local dev overrides (hot reload, bind mounts)

## Quick start (local dev)

Prerequisites: Docker, Docker Compose v2, Python 3.12, Node 20+.

```bash
cp .env.example .env
# Fill in MINIO_*, GITHUB_TOKEN (optional), ANTHROPIC_API_KEY (optional)

docker compose -f docker-compose.yaml -f docker-compose.dev.yaml up --build
```

- API:    http://localhost:8000
- Web:    http://localhost:3000
- Docs:   http://localhost:8000/docs (auto-generated OpenAPI)
- Health: http://localhost:8000/api/health

## Production deployment

Runs on the `ci-logs` Brev instance alongside MinIO. Caddy reverse-proxies
`https://dashboard.34-93-45-118.nip.io` to the two containers.

See [DEPLOY.md](DEPLOY.md) for Brev-specific steps.

## Data flow

```
MinIO (sglang-ci-logs bucket)
    │
    │ every 5 min: list new objects
    ▼
Ingester (APScheduler inside API container)
    │ parse result JSONs + enrich via GitHub API
    ▼
SQLite (/data/dashboard.db, 90-day retention)
    │
    ▼
FastAPI (/api/*) ◄─── Next.js frontend
```

## Milestones

- **M1 (current):** ingester + API + home/runs/run-detail pages
- **M2:** compare tool, regression detection, commit tracker, dark mode
- **M3:** AI summaries, annotations, keyboard shortcuts, real auth
- **M4+:** bisect helper, per-author scorecard, Slack integration

## Links

- Design doc: DLR-5301
- Parent: DLR-5069 (GB200 nightly pipeline)
- MinIO logs: https://minio.34-93-45-118.nip.io
