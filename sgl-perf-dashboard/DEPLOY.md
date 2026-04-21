# Deployment on the `ci-logs` Brev instance

This dashboard runs alongside MinIO + Caddy on the same Brev box. Caddy
terminates TLS for `dashboard.34-93-45-118.nip.io` and reverse-proxies to
the two containers on the shared `minio-net` Docker network.

## Prerequisites on the box

- MinIO + Caddy are already running (see `/home/ubuntu/minio/`).
- The Docker network `minio_minio-net` (created by MinIO's compose) exists.
- Brev "Exposed ports" already allow 80/443 to the public internet.

## One-time setup

```bash
ssh ci-logs
cd ~
git clone https://github.com/sgl-project/sglang.git   # or already on disk
cd sglang/sgl-perf-dashboard

# Copy env and fill in values (see .env.example)
cp .env.example .env
chmod 600 .env
vim .env   # fill MINIO_ACCESS_KEY / MINIO_SECRET_KEY / GITHUB_TOKEN

# Ensure the Docker network MinIO uses is addressable under the name minio-net.
# MinIO's compose creates it as `minio_minio-net`. Create an alias:
docker network create --driver bridge minio-net 2>/dev/null || true
# Alternative: edit docker-compose.yaml to reference `minio_minio-net` directly
```

## Caddy addition

Append to `/home/ubuntu/minio/Caddyfile`:

```caddyfile
dashboard.34-93-45-118.nip.io {
    @api path /api/*
    reverse_proxy @api dashboard-api:8000
    reverse_proxy dashboard-web:3000
}
```

Reload Caddy:
```bash
cd /home/ubuntu/minio
docker compose exec caddy caddy reload --config /etc/caddy/Caddyfile
# or simply:
docker compose restart caddy
```

## Start the dashboard

```bash
cd ~/sglang/sgl-perf-dashboard
docker compose up -d --build

# Watch logs
docker compose logs -f
```

Verify:
```bash
curl -I https://dashboard.34-93-45-118.nip.io/api/health
```

## Upgrade

```bash
cd ~/sglang
git pull
cd sgl-perf-dashboard
docker compose up -d --build
```

## Backup / migrate

SQLite lives at `./dashboard-data/dashboard.db`. Back up with:
```bash
cp dashboard-data/dashboard.db dashboard-data/dashboard.db.$(date +%Y%m%d)
```

To rebuild from scratch: stop containers, delete `dashboard-data/dashboard.db`,
restart. Ingester will re-scan MinIO from the beginning (~10 min for 90 days).
