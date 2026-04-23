# Deployment on the `ci-logs` Brev instance

This dashboard runs alongside MinIO + Caddy on the same Brev box. The
dashboard and MinIO console are exposed to the team via Cloudflare tunnels
(`sgl-dashboard-khjfeoysf.brevlab.com`, `sgl-ci-logs-khjfeoysf.brevlab.com`),
gated by Cloudflare Access. The MinIO S3 API used by CI keeps using the
direct path through Caddy on the Brev public IP.

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

## Tunnel + host-port setup

The dashboard-web and MinIO containers must bind to `127.0.0.1` on the host
so Brev's tunnel agent can reach them:

- `docker-compose.yaml` → `dashboard-web.ports`: `127.0.0.1:3000:3000`
- `~/minio/docker-compose.yaml` → `minio.ports`: `127.0.0.1:9001:9001`

Then create two Brev tunnels from the instance UI:
- `sgl-dashboard-*` → internal port 3000
- `sgl-ci-logs-*`   → internal port 9001

Enable Cloudflare Access on both, with an `@nvidia.com` policy.

## Caddy (for CI S3 API only)

Caddy still fronts the MinIO S3 API at `https://minio.<host>.nip.io` for CI
uploads. No dashboard vhost is needed — Brev's tunnel bypasses Caddy for
human-facing traffic.

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
curl -I https://sgl-dashboard-khjfeoysf.brevlab.com/api/health
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
