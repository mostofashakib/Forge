# Forge deployment

Forge runs as one Docker Compose stack on a single VM, reached only over
Tailscale. This file is the contract for how the services connect. Change the
wiring map here first, then `wiring-check.sh`, then the code.

## Decisions

Pending. Each answer gets the date and who decided.

| Decision | Answer | Decided by | Date |
|---|---|---|---|
| Cloud provider | pending | | |
| Access through Tailscale only, no public port | pending | | |
| SQLite on the VM disk | pending | | |
| LLM provider | pending | | |
| Who gets alert emails | pending | | |


## Stack

Everything except the VM is open source or free on every provider.

| Need | Choice | Cost |
|---|---|---|
| VM | See the next table | Free on Oracle, about $100 a month elsewhere |
| Containers | Docker Engine + Docker Compose | Open source |
| HTTPS and routing | Caddy | Open source |
| Private access | Tailscale (Personal plan) | Free for 3 users and 100 devices. Fully open-source alternative: Headscale |
| Queue and progress pub/sub | Redis (or Valkey, its open-source fork) | Open source |
| LLM, free option | Ollama (Compose profile `ollama`) | Open source. Hosted providers need their own key |
| Image registry | GitHub Container Registry (GHCR) | Free for public images, free storage quota for private ones |
| CI/CD | GitHub Actions | Free minutes (unlimited for public repos) |
| Database backup | Litestream to the provider's object storage | Open source |
| File backup | restic to the same bucket | Open source |
| Alerts | Healthchecks.io (heartbeat model) | Free for 20 checks, also open source and self-hostable |
| Log viewing | Dozzle | Open source |

## Cloud provider options

The design is the same on every provider: one VM running Docker Compose, reached
only over Tailscale. Only the rows below change.

| | Oracle (default) | AWS | Azure | GCP |
|---|---|---|---|---|
| VM | Ampere A1, 4 OCPU, 24 GB (arm64) | `t4g.xlarge`, 4 vCPU, 16 GB (arm64) | `Standard_D4ps_v5`, 4 vCPU, 16 GB (arm64) | `t2a-standard-4`, 4 vCPU, 16 GB (arm64), or `e2-standard-4` (x86) |
| VM cost | $0 (Always Free) | About $100 a month on demand | About $110 a month on demand | About $100 to $115 a month on demand |
| Free tier fits Forge? | Yes | No. Free-tier VMs have 1 to 2 GB RAM. New-account credits cover a few months | No. Free B1s has 1 GB. $200 credit for the first 30 days | No. Always Free e2-micro has 1 GB |
| Backup storage | Cloudflare R2 (10 GB free, no egress fees) or OCI Object Storage | S3 | Blob Storage (not S3 API, needs the Azure backends in Litestream and restic) | Cloud Storage with HMAC keys (S3 API) |
| Network firewall that closes public SSH | VCN security list | Security group | Network security group | VPC firewall rule |
| Native alternative to Tailscale | Bastion service | SSM Session Manager or Client VPN | Azure Bastion | IAP TCP forwarding |
| Native registry (optional, GHCR works everywhere) | OCIR | ECR | ACR | Artifact Registry |
| Startup credits | Oracle for Startups | AWS Activate | Microsoft for Startups | Google for Startups Cloud Program |

Notes:
- Prices are list prices in US regions and change. Check the provider's calculator before creating the VM. Spot or preemptible VMs cost much less but get stopped without warning, which kills running rollouts. One- or three-year commitments cut the on-demand price by roughly a third or more.
- Forge needs about 16 GB of RAM because every environment and rollout runs in its own container. 8 GB VMs run out of memory during a 6-sandbox rollout.
- Keep Healthchecks.io on any provider. It watches from outside your cloud, so it still emails you when the provider's own region or monitoring has an outage. The native monitors (CloudWatch, Azure Monitor, Cloud Monitoring) work as a second layer.
- Oracle's free Ampere capacity sells out in popular regions. Retry, or pick a quieter home region. If it stays unavailable, any provider above works with the same plan.

## Compose services

| Service | Image | Network | Mounts |
|---|---|---|---|
| `caddy` | `caddy:2` | host | `deploy/Caddyfile`, `caddy-data`, `/etc/forge/certs` read-only |
| `frontend` | `forge-frontend` | host | none |
| `api` | `forge-backend` (API command) | host | `forge-data`, `generated-envs`, `model-cache`, `docker.sock` |
| `worker` | `forge-backend` (worker command) | host | `forge-data`, `generated-envs`, `model-cache`, `docker.sock` |
| `redis` | `redis:7` or `valkey/valkey:8` | host, bound to `127.0.0.1`, password required | `redis-data` |
| `litestream` | `litestream/litestream` | default | `forge-data`, `deploy/litestream.yml` |
| `backup` | `restic/restic` with `deploy/backup.sh` | default | `generated-envs` read-only |
| `heartbeat` | `curlimages/curl` with `deploy/heartbeat.sh` | host | `forge-data` read-only (for the disk check) |
| `dozzle` | `amir20/dozzle` | host, bound to `127.0.0.1:9999` | `docker.sock` read-only |
| `ollama` (profile `ollama`) | `ollama/ollama` | host, bound to `127.0.0.1:11434` | `ollama-models` |

Host networking for `api`, `worker`, `frontend`, and `caddy` keeps environment
gateways on `127.0.0.1`, the same way `run.sh` works today.

## The wiring map

| ID | From | To | Address | Credential | Tested by running |
|---|---|---|---|---|---|
| W1 | Teammate browser | Caddy | `https://forge.<tailnet>.ts.net:443` over the tailnet | Tailscale ACL, `tailscale cert` | `curl` from a tailnet device (smoke.sh) |
| W2 | Caddy | frontend | `127.0.0.1:3000` | none (loopback) | `curl https://<host>/` returns the app shell |
| W3 | Caddy | api, HTTP and WebSocket | `127.0.0.1:8000`, path `/api/*` | none (loopback) | `curl /api/health` and a `wss://` handshake through Caddy |
| W4 | Caddy | Dozzle | `127.0.0.1:9999`, path `/logs` | Tailscale ACL | `curl https://<host>/logs/` returns 200 |
| W5 | Browser JS | api | same origin, `NEXT_PUBLIC_API_URL=https://<host>` | `CORS_ORIGINS` matches the host | a preflight `OPTIONS` with `Origin: https://<host>` is allowed |
| W6 | api | Redis | `REDIS_URL` on `127.0.0.1:6379` | `FORGE_REDIS_PASSWORD` | `docker compose exec api` runs a Redis `PING` |
| W7 | worker | Redis | same as W6 (Celery broker, progress pub/sub) | `FORGE_REDIS_PASSWORD` | `celery inspect ping` returns `pong` from the worker |
| W8 | api, worker | SQLite | `/data/forge.db` on volume `forge-data` | file permissions (uid `forge`) | both containers write and read a row |
| W9 | api, worker | Docker daemon | `/var/run/docker.sock` | `docker` group | `docker compose exec worker docker version` |
| W10 | worker | environment gateways | `sandbox_url(port)` from `FORGE_SANDBOX_HOST` | none | start a premade environment and `GET` its gateway |
| W11 | frontend `/api/proxy` route (general sandboxes, HTTP), and api `/api/sandbox/{env}/browser/` route (browser sandboxes, HTTP and WebSocket) | environment gateways | `FORGE_SANDBOX_HOST:{port}` | none | `GET https://<host>/api/proxy/<env>/` returns the environment's page, and a `wss://` handshake to `/api/sandbox/<env>/browser/api/websockets` succeeds |
| W12 | api, worker | LLM provider | Ollama at `OLLAMA_BASE_URL`, or the hosted provider's HTTPS API | provider key via `forge/credentials.py` | `/api/health` reports `llm: ok` |
| W13 | Litestream | `forge.db`, then the backup bucket | volume `forge-data`, `forge-backups/<deploy>/db` | key scoped to `forge-backups` | `litestream snapshots` lists a snapshot under 1 h old |
| W14 | backup | volume `generated-envs`, then the backup bucket | read-only mount, `forge-backups/<deploy>/files` | same key, `RESTIC_PASSWORD` | `restic snapshots` lists one under 25 h old |
| W15 | heartbeat | api, then Healthchecks.io | `http://127.0.0.1:8000/api/health`, `FORGE_HC_HEALTH_URL` | ping URL in the env file | the Healthchecks.io API shows `forge-health` pinged in the last 10 min |
| W16 | backup | Healthchecks.io | `FORGE_HC_BACKUP_URL` | ping URL in the env file | the Healthchecks.io API shows `forge-backup` pinged in the last 25 h |
| W17 | Dozzle | Docker daemon | `/var/run/docker.sock`, read-only | `docker` group | `/logs` lists the api container |
| W18 | GitHub Actions | GHCR, then the VM | `ghcr.io/mostofashakib/forge-*`, Tailscale SSH | `GITHUB_TOKEN` (packages), Tailscale OAuth client | the latest deploy run shows push, SSH, and a post-deploy smoke pass |
| W19 | VM | GHCR | `docker compose pull` | public images or `GHCR_READ_TOKEN` | `docker compose pull` exits 0 |

**Forbidden connections**

| ID | Must fail | Tested by |
|---|---|---|
| X1 | The internet reaching any port on the VM | `nmap -Pn` against the public IP from a non-tailnet machine reports every port filtered |
| X2 | Redis without the password | `redis-cli -h 127.0.0.1 PING` returns `NOAUTH` |
| X3 | Redis, api, frontend, or Dozzle on a non-loopback address | `ss -tlnp` shows them on `127.0.0.1` only |
| X4 | Docker socket inside `caddy`, `frontend`, `redis`, `litestream`, `backup`, or `heartbeat` | `docker inspect` mounts for each one have no `docker.sock` |
| X5 | A browser origin other than the Forge host calling the api | a preflight with `Origin: https://evil.example` gets no `Access-Control-Allow-Origin` |

## What happens when something breaks

Every failure travels the same chain, so one alert path covers every service.

```
dependency fails -> /api/health returns 503 naming it -> Docker marks api unhealthy
               -> heartbeat sends /fail -> Healthchecks.io emails the team
container exits -> restart: unless-stopped brings it back -> health returns to 200 -> next ping clears the alert
backup fails   -> backup service pings forge-backup/fail -> Healthchecks.io emails the team
deploy fails   -> pipeline rolls back and pings forge-health/fail -> Healthchecks.io emails the team
```

| Service | Reported by | Alert check |
|---|---|---|
| Redis | `/api/health` `checks.redis` | `forge-health` |
| SQLite volume | `/api/health` `checks.database` | `forge-health` |
| Docker daemon | `/api/health` `checks.docker` | `forge-health` |
| worker | `/api/health` `checks.worker` | `forge-health` |
| LLM provider | `/api/health` `checks.llm` | `forge-health` |
| Litestream | `/api/health` `checks.backup` | `forge-health` |
| restic job | its own exit code | `forge-backup` |
| api | heartbeat cannot reach health | `forge-health` |
| disk | heartbeat sees the data volume over 85% full | `forge-health` |
| Caddy | smoke.sh only (known gap: the heartbeat calls the api directly) | none |
