"""Secret detectors catch credentials without flagging ordinary identifiers."""
import pytest

from forge.safety.secrets import detect_secrets


@pytest.mark.parametrize("text,value", [
    ("key sk-proj-AbCdEfGhIjKlMnOpQrStUvWx123456 here", "sk-proj-AbCdEfGhIjKlMnOpQrStUvWx123456"),
    ("ANTHROPIC_API_KEY=sk-ant-api03-AbCdEfGhIjKlMnOpQrStUv_wxyz", "sk-ant-api03-AbCdEfGhIjKlMnOpQrStUv_wxyz"),
    ("aws AKIAIOSFODNN7EXAMPLE", "AKIAIOSFODNN7EXAMPLE"),
    ("gh ghp_" + "a" * 36, "ghp_" + "a" * 36),
    ("pat github_pat_11ABCDEFG0123456789_abcdefghijklmnop", "github_pat_11ABCDEFG0123456789_abcdefghijklmnop"),
    ("slack xoxb-1234567890-abcdefghij", "xoxb-1234567890-abcdefghij"),
    ("Authorization: Bearer eyJhbGciOi.payload.sig", "Bearer eyJhbGciOi.payload.sig"),
    ("db postgres://admin:hunter2@db.internal/app", "admin:hunter2"),
])
def test_each_secret_kind_is_found(text, value):
    assert [span.value for span in detect_secrets(text)] == [value]


def test_a_pem_private_key_block_is_found_whole():
    pem = "-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIBAAKCAQEA\nabc\n-----END RSA PRIVATE KEY-----"
    (span,) = detect_secrets(f"key:\n{pem}\nend")
    assert span.value == pem
    assert span.kind == "secret"


@pytest.mark.parametrize("text", [
    "task-sk-ticket closed",
    "commit deadbeefcafe",
    "use a Bearer token here",
    "see https://example.com/@user",
    "AKIA is a prefix",
    "a public key -----BEGIN PUBLIC KEY----- block",
])
def test_ordinary_identifiers_are_not_secrets(text):
    assert detect_secrets(text) == []
