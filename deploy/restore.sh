#!/usr/bin/env bash
# Restore one deployment's backups into fresh volumes under another Compose project.
#
#   deploy/restore.sh <from-deploy-name> <compose-project> [env-file]
#   deploy/restore.sh vm forge-restore deploy/.env.production
#
# Reads only. It never writes to the source deployment's backup prefix. It refuses
# to touch a project whose volumes already exist. Afterwards, start the restored
# stack with the command it prints. Stop any other Forge stack on this machine
# first: the services share the host's ports.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FROM="${1:?usage: restore.sh <from-deploy-name> <compose-project> [env-file]}"
PROJECT="${2:?usage: restore.sh <from-deploy-name> <compose-project> [env-file]}"
ENV_FILE="${3:-$DIR/.env.production}"

env_value() { grep -E "^$1=" "$ENV_FILE" | tail -1 | cut -d= -f2- | sed -e 's/^"//' -e 's/"$//'; }
say() { echo "[restore] $*"; }

DATA="${PROJECT}_forge-data"
ENVS="${PROJECT}_generated-envs"
for volume in "$DATA" "$ENVS"; do
    if docker volume inspect "$volume" >/dev/null 2>&1; then
        echo "volume $volume already exists. Restore only into a fresh project." >&2
        exit 1
    fi
    docker volume create --label "com.docker.compose.project=$PROJECT" \
        --label "com.docker.compose.volume=${volume#"${PROJECT}_"}" "$volume" >/dev/null
done
# The api and Litestream run as uid 10001.
docker run --rm -v "$DATA:/data" alpine:3.22 chown 10001:10001 /data

S3=(-e "LITESTREAM_ACCESS_KEY_ID=$(env_value BACKUP_S3_ACCESS_KEY_ID)"
    -e "LITESTREAM_SECRET_ACCESS_KEY=$(env_value BACKUP_S3_SECRET_ACCESS_KEY)"
    -e "BACKUP_S3_ENDPOINT=$(env_value BACKUP_S3_ENDPOINT)"
    -e "BACKUP_S3_REGION=$(env_value BACKUP_S3_REGION)"
    -e "BACKUP_S3_BUCKET=$(env_value BACKUP_S3_BUCKET)"
    -e "FORGE_DEPLOY_NAME=$FROM")

say "database from $(env_value BACKUP_S3_BUCKET)/$FROM/db"
docker run --rm -u 10001:10001 "${S3[@]}" -v "$DATA:/data" -v "$DIR/litestream.yml:/etc/litestream.yml:ro" \
    litestream/litestream:0.5.17 restore -config /etc/litestream.yml /data/forge.db

say "files from $(env_value BACKUP_S3_BUCKET)/$FROM/files"
docker run --rm -v "$ENVS:/generated_envs" \
    -e "RESTIC_REPOSITORY=s3:$(env_value BACKUP_S3_ENDPOINT)/$(env_value BACKUP_S3_BUCKET)/$FROM/files" \
    -e "RESTIC_PASSWORD=$(env_value RESTIC_PASSWORD)" \
    -e "AWS_ACCESS_KEY_ID=$(env_value BACKUP_S3_ACCESS_KEY_ID)" \
    -e "AWS_SECRET_ACCESS_KEY=$(env_value BACKUP_S3_SECRET_ACCESS_KEY)" \
    -e "AWS_DEFAULT_REGION=$(env_value BACKUP_S3_REGION)" \
    restic/restic:0.19.1 restore latest --host forge --target /

say "integrity check"
BACKEND_IMAGE="$(env_value FORGE_IMAGE_REPO)/forge-backend:$(env_value FORGE_IMAGE_TAG)"
result=$(docker run --rm -v "$DATA:/data" "$BACKEND_IMAGE" sqlite3 /data/forge.db "PRAGMA integrity_check")
say "PRAGMA integrity_check: $result"
[[ "$result" == "ok" ]] || exit 1

say "restored. Start it with:"
echo "  FORGE_DEPLOY_NAME=restore-drill docker compose -p $PROJECT -f $DIR/docker-compose.yml --env-file $ENV_FILE up -d"
