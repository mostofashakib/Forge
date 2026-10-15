"""module_name_for: the Python package name a generated environment's code lives under."""
from __future__ import annotations

import pytest

from forge.paths import module_name_for, validate_identifier


@pytest.mark.parametrize(
    ("env_name", "module"),
    [
        ("todo_app", "todo_app"),
        ("my-env", "my_env"),
        ("crm-v2-beta", "crm_v2_beta"),
        ("2fa-portal", "env_2fa_portal"),
        ("class", "class_env"),
        ("import", "import_env"),
    ],
)
def test_every_valid_env_name_becomes_an_importable_module_name(env_name, module):
    assert module_name_for(env_name) == module
    assert validate_identifier(module_name_for(env_name)) == module


@pytest.mark.parametrize("env_name", ["", "../etc", "has space", "-leading"])
def test_rejects_a_name_the_api_would_never_accept(env_name):
    with pytest.raises(ValueError, match="letters, digits, underscores, and hyphens"):
        module_name_for(env_name)
