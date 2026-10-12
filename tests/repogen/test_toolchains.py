"""Toolchains detect a repo's stack, choose its commands, and read per-test results."""
import json

import pytest

from forge.repogen.toolchains import NodeToolchain, PythonToolchain, detect_toolchain


def _write(root, files):
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return root


PKG_JEST = json.dumps({"scripts": {"test": "jest"}, "devDependencies": {"jest": "^29.0.0"}})
PKG_VITEST = json.dumps({"scripts": {"test": "vitest"}, "devDependencies": {"vitest": "^1.6.0"}})


@pytest.mark.parametrize("files,expected", [
    ({"pyproject.toml": "[project]\nname='x'\n"}, "python"),
    ({"setup.py": "from setuptools import setup\n"}, "python"),
    ({"requirements.txt": "requests\n"}, "python"),
    ({"package.json": PKG_JEST}, "node"),
    ({"package.json": PKG_VITEST}, "node"),
    ({"package.json": json.dumps({"scripts": {"test": 'echo "Error: no test specified" && exit 1'}})}, None),
    ({"package.json": json.dumps({"scripts": {"test": "mocha"}, "devDependencies": {"mocha": "1"}})}, None),
    ({"package.json": "not json"}, None),
    ({"main.go": "package main"}, None),
    ({}, None),
])
def test_detection_matrix(tmp_path, files, expected):
    toolchain = detect_toolchain(_write(tmp_path, files), language=None)
    assert (toolchain.name if toolchain else None) == expected


def test_the_repo_language_decides_when_both_stacks_are_present(tmp_path):
    _write(tmp_path, {"pyproject.toml": "[project]\nname='x'\n", "package.json": PKG_JEST})
    assert detect_toolchain(tmp_path, language="TypeScript").name == "node"
    assert detect_toolchain(tmp_path, language="Python").name == "python"


def test_python_installs_the_first_test_extra_and_requirements(tmp_path):
    _write(tmp_path, {
        "pyproject.toml": "[project]\nname='x'\n[project.optional-dependencies]\ndev=['ruff']\ntests=['pytest']\n",
        "requirements-dev.txt": "black\n",
    })
    commands = PythonToolchain().install_commands(tmp_path)
    assert 'pip install --no-cache-dir -e ".[tests]"' in commands
    assert "pip install --no-cache-dir -r requirements-dev.txt" in commands
    assert commands[-1] == "pip install --no-cache-dir pytest"


def test_python_without_extras_installs_the_package(tmp_path):
    _write(tmp_path, {"setup.py": "from setuptools import setup\nsetup()\n"})
    assert "pip install --no-cache-dir -e ." in PythonToolchain().install_commands(tmp_path)


def test_python_requirements_only_repo_skips_the_package_install(tmp_path):
    _write(tmp_path, {"requirements.txt": "flask\n"})
    commands = PythonToolchain().install_commands(tmp_path)
    assert not any("-e ." in command for command in commands)
    assert "pip install --no-cache-dir -r requirements.txt" in commands


@pytest.mark.parametrize("lockfile,command", [
    ("package-lock.json", "npm ci"),
    ("pnpm-lock.yaml", "corepack enable && pnpm install --frozen-lockfile"),
    ("yarn.lock", "corepack enable && yarn install --frozen-lockfile"),
    (None, "npm install"),
])
def test_node_install_follows_the_lockfile(tmp_path, lockfile, command):
    files = {"package.json": PKG_JEST}
    if lockfile:
        files[lockfile] = "{}"
    assert NodeToolchain().install_commands(_write(tmp_path, files)) == [command]


def test_test_commands_write_a_report(tmp_path):
    assert PythonToolchain().test_command("/tmp/r.xml", tmp_path) == (
        "python -m pytest -q -p no:cacheprovider --junitxml=/tmp/r.xml"
    )
    _write(tmp_path, {"package.json": PKG_VITEST})
    node = NodeToolchain()
    assert node.test_command("/tmp/r.json", tmp_path) == "npx vitest run --reporter=json --outputFile=/tmp/r.json"
    _write(tmp_path, {"package.json": PKG_JEST})
    assert node.test_command("/tmp/r.json", tmp_path) == "npx jest --ci --json --outputFile=/tmp/r.json"


JUNIT = """<?xml version="1.0"?>
<testsuites><testsuite name="pytest">
  <testcase classname="tests.test_app" name="test_ok"/>
  <testcase classname="tests.test_app" name="test_fail"><failure message="boom"/></testcase>
  <testcase classname="tests.test_app" name="test_err"><error message="x"/></testcase>
  <testcase classname="tests.test_app" name="test_skip"><skipped/></testcase>
  <testcase classname="tests.test_app" name="test_param[a-1]"/>
</testsuite></testsuites>"""


def test_junit_reports_count_only_clean_passes():
    assert PythonToolchain().parse_report(JUNIT) == {
        "tests.test_app::test_ok": True,
        "tests.test_app::test_fail": False,
        "tests.test_app::test_err": False,
        "tests.test_app::test_skip": False,
        "tests.test_app::test_param[a-1]": True,
    }


def test_jest_style_reports_use_repo_relative_ids():
    report = json.dumps({"testResults": [{
        "name": "/repo/src/math.test.ts",
        "assertionResults": [
            {"fullName": "add sums", "status": "passed"},
            {"fullName": "add overflows", "status": "failed"},
            {"fullName": "add later", "status": "pending"},
        ],
    }]})
    assert NodeToolchain().parse_report(report) == {
        "src/math.test.ts::add sums": True,
        "src/math.test.ts::add overflows": False,
        "src/math.test.ts::add later": False,
    }


@pytest.mark.parametrize("parser,text", [(PythonToolchain().parse_report, "<not xml"), (NodeToolchain().parse_report, "{")])
def test_an_unreadable_report_raises(parser, text):
    with pytest.raises(ValueError):
        parser(text)
