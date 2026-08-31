from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_local_runner_requires_redis_authentication():
    script = (PROJECT_ROOT / "run.sh").read_text()

    assert "requirepass %s" in script
    assert 'chmod 600 "$REDIS_CONF"' in script
    assert "redis://:${REDIS_PASSWORD}@127.0.0.1:6379/0" in script
    assert 'export REDIS_URL="$FORGE_LOCAL_REDIS_URL"' in script
    assert 'export CELERY_BROKER_URL="$FORGE_LOCAL_REDIS_URL"' in script
    assert 'export CELERY_RESULT_BACKEND="$FORGE_LOCAL_REDIS_URL"' in script


def test_local_runner_does_not_expose_password_in_redis_process_arguments():
    script = (PROJECT_ROOT / "run.sh").read_text()

    assert 'redis-server "$REDIS_CONF"' in script
    assert "redis-server --requirepass" not in script
