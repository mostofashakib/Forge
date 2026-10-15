# Deployment progress

Evidence for each verify step in the deployment plan: the step ID, the exact
command, and the last lines of its output. Secrets never appear here.

All Phase 1 work happens on the branch `deploy/phase-1` (one branch, by the
user's request, in place of the plan's stacked milestone branches).

## Phase 1. Everything the agent does alone

### A3. deploy/README.md lists every wiring ID

`grep -oE '\b[WX][0-9]+\b' deploy/README.md | sort -u -V`
```
W1 W2 W3 W4 W5 W6 W7 W8 W9 W10 W11 W12 W13 W14 W15 W16 W17 W18 W19 X1 X2 X3 X4 X5
```

### A6. Health endpoint

Tests: `uv run pytest -q tests/backend/test_health_api.py` gives `21 passed`. With the
router unregistered, 8 endpoint tests fail. Full suite: `2718 passed, 2 skipped`.

Live checks ran on an isolated stack (Redis 6391, API 8191, its own Celery worker,
a scratch database) because ports 8000 and 3000 belong to another project on
this laptop, and `run.sh` would kill them.

```
== all up
{"status":"ok","checks":{"database":"ok","redis":"ok","docker":"ok","worker":"ok","llm":"ok","backup":"disabled"}} 200
== redis down
{"status":"degraded","checks":{..."redis":"Error 61 connecting to 127.0.0.1:6391. Connection refused.",...,"worker":"Error 61 connecting to 127.0.0.1:6391. Connection refused.",...}} 503
== redis back (no API restart)
{"status":"ok",...} 200
== worker down
{"status":"degraded","checks":{...,"worker":"no Celery worker answered a ping",...}} 503
== worker back
{"status":"ok",...} 200
== OLLAMA_BASE_URL pointing at a closed port
"llm":"could not list models from http://127.0.0.1:11999/api/tags: [Errno 61] Connection refused"
== curl -s localhost:8191/api/health | grep -Ei 'redis://|password|sk-|key='
(no output, grep exit 1)
```

### A9. One setting for the sandbox host

`grep -rnE "f[\"']https?://(localhost|127\.0\.0\.1):\{" forge backend --include='*.py' | grep -v tests/`
prints nothing (grep exit 1). The plan listed four call sites. A fifth,
`task_factory_targets.py:181` (the browser CDP URL), was also moved to `sandbox_url`.

New tests: `10 passed`. Reverting `sandbox_build_tasks.py:352` to a hand-built URL fails
the architecture test with `'backend/app/worker/sandbox_build_tasks.py:352'`, and
restoring it passes. One existing test expected `http://localhost:{port}` for the CDP
URL and now expects `http://127.0.0.1:{port}`, the published gateway address.
Full suite: `2728 passed, 2 skipped`.

Live, on the isolated stack: `POST /api/sandbox/ {"env_name":"m2-gmail","env_type":"premade:gmail"}`
reached `running` on port 50646. A 1-episode `random` agent run finished
`completed 1`, episode `completed 4 steps, reward 0.7, submitted`. The worker log shows
19 requests to `http://127.0.0.1:50646` and none to `localhost:<sandbox port>`.

### A14. Sandbox pages through the proxy

Finding: the browser sandbox's desktop is Selkies, which opens its WebSocket at
`<page directory>/api/websockets`. The Next.js `/api/proxy` route carries only HTTP,
so browser sandboxes now go through a new API route,
`/api/sandbox/{env_name}/browser/{path}`, that carries HTTP and WebSocket traffic and
reaches only the port the database records for that sandbox. General sandboxes keep
`/api/proxy`, now with its upstream host from `FORGE_SANDBOX_HOST`.

Consequence for the Caddyfile (A25): `/api/proxy/*` must route to the frontend, ahead
of the `/api/*` rule that routes to the API.

`grep -rn 'localhost:\${' frontend/app frontend/components frontend/lib` prints nothing.
`npm --prefix frontend test`: `pass 9, fail 0`. `npm --prefix frontend run build`: passes.
`uv run pytest -q tests/backend/test_sandbox_browser_proxy.py`: `10 passed`. Full suite:
`2738 passed, 2 skipped`.

Live, headless Chrome on the isolated stack (frontend 3191, API 8191):
```
browser sandbox page (m3-browser, raw port 50717)
hosts contacted: ['localhost:3191', 'localhost:8191']
proxied: http://localhost:8191/api/sandbox/m3-browser/browser/assets/selkies-core-BbKps5RD.js (and 5 more)
websocket: ws://localhost:8191/api/sandbox/m3-browser/browser/api/websockets frames received: 137
failures: none
(the screenshot shows the Chromium desktop rendered in the iframe)

gmail sandbox page (m2-gmail, raw port 50915)
hosts contacted: ['fonts.googleapis.com', 'fonts.gstatic.com', 'localhost:3191', 'localhost:8191']
proxied: http://localhost:3191/api/proxy/m2-gmail/ui, .../forge/state
failures: none
Compose and Send inside the iframe: sent_count 5 -> 6, read back through /api/proxy/m2-gmail/forge/state
```

From the laptop's LAN address (frontend bound to all interfaces):
```
$ curl -s -o /dev/null -w '%{http_code}' http://192.168.1.202:3191/api/proxy/m2-gmail/ui
200
$ curl -s http://192.168.1.202:3191/api/proxy/m2-gmail/ui | grep -o "<title>[^<]*</title>"
<title>Gmail</title>
```
(The first attempt timed out while the macOS application firewall held the connection.
It passed once Node was allowed through.)

### A18. Production startup checks

`uv run pytest -q tests/backend/test_production_checks.py`: `21 passed`. Full suite:
`2759 passed, 2 skipped`.

Refinement: Forge reads only `REDIS_URL`, so the check requires that `REDIS_URL`
carries a password (the problem line names `FORGE_REDIS_PASSWORD`), in place of
requiring a `FORGE_REDIS_PASSWORD` variable Forge never reads.

```
$ FORGE_ENV=production FORGE_SCRUB_KEY= ... uvicorn backend.app.main:app
[forge] FORGE_SCRUB_KEY is not set. It keys PII pseudonyms in internet and human data.
exit 1
$ FORGE_ENV=production FORGE_SCRUB_KEY= ... celery -A backend.app.worker.celery_app worker
[forge] FORGE_SCRUB_KEY is not set. It keys PII pseudonyms in internet and human data.
exit 1
$ FORGE_ENV=production (every value set) uvicorn ... --port 8193; curl /api/health
{"status":"ok","checks":{"database":"ok","redis":"ok","docker":"ok","worker":"ok","llm":"ok","backup":"disabled"}} 200
```
With `FORGE_ENV` unset, the API starts as before (the isolated stack on 8191 runs that way).

`deploy/.env.production.example` added. `.gitignore` gains `!deploy/.env.production.example`,
and `git check-ignore` still ignores `deploy/.env.production` and `deploy/.env.vm`.

### A23. Container images

```
$ docker buildx build --platform linux/amd64,linux/arm64 -f docker/forge/Dockerfile .      -> exit 0
$ docker buildx build --platform linux/amd64,linux/arm64 frontend (NEXT_PUBLIC_API_URL arg) -> exit 0
$ docker run --rm forge-backend:dev id -u     -> 10001
$ docker run --rm forge-frontend:dev id -u    -> 1000
$ docker run --rm forge-backend:dev python -c "import backend.app.main"   -> exit 0
$ docker run --rm forge-backend:dev sh -c 'ls -a / /app | grep -E "\.env$|forge\.db"'   -> no output
  (a full `find / -xdev` for .env, .env.*, forge.db* in both images also finds nothing)
$ docker history --no-trunc forge-backend:dev | grep -Ei 'api_key|token|password='    -> no output
$ docker history --no-trunc forge-frontend:dev | grep -Ei 'api_key|token|password='   -> no output
$ docker images   -> forge-backend:dev 2.77GB, forge-frontend:dev 295MB
```
The first build was 4.6 GB and 405 MB. The uv download cache had landed in the
dependency layer (fixed with a BuildKit cache mount), and the Debian Node base was
replaced with Alpine. Linux PyTorch now comes from the CPU wheel index (approved by
the user), pinned to 2.13.0 on every platform, which removed 15 CUDA packages.

Models: loading all three into an empty volume took 112 s and 1.5 GB. Documented in
`deploy/README.md` rather than baked into the image.

## Phase 2. The full system on the laptop

## Phase 3. The VM

## Phase 4. Push-to-deploy
