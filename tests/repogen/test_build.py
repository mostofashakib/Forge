"""Building a repo environment: image, two baseline runs, and the stable test set."""
import json

import pytest

from forge.repogen.build import build_repo_environment, render_dockerfile, repo_image_tag
from forge.repogen.toolchains import PythonToolchain

JUNIT = '<testsuites><testsuite>{}</testsuite></testsuites>'


def _junit(results):
    cases = "".join(
        f'<testcase classname="t" name="{name}"/>' if passed
        else f'<testcase classname="t" name="{name}"><failure/></testcase>'
        for name, passed in results.items()
    )
    return JUNIT.format(cases)


class FakeDocker:
    def __init__(self, fail_build=None):
        self.fail_build = fail_build
        self.runs = []
        self.discarded = []

    def run(self, *args, timeout=900):
        self.runs.append(args)
        if args[0] == "build" and self.fail_build:
            raise RuntimeError(self.fail_build)

    def succeeds(self, *args):
        return True

    def discard(self, *args):
        self.discarded.append(args)


class FakeSandbox:
    def __init__(self, reports):
        self.reports = list(reports)
        self.commands = []

    def run(self, image, command, timeout_s):
        self.commands.append((image, command))
        return self.reports.pop(0)


def _repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname='x'\n")
    (root / "Dockerfile").write_text("FROM scratch\n")
    (root / ".dockerignore").write_text("tests\n")
    return root


def _build(tmp_path, docker, sandbox, min_stable=2, language="Python"):
    pulled = []
    result = build_repo_environment(
        "acme_widgets", _repo(tmp_path), language, docker=docker, sandbox=sandbox,
        min_stable_tests=min_stable, pull=pulled.append,
    )
    return result, pulled


def test_render_dockerfile_installs_with_faketime_and_copies_the_repo(tmp_path):
    text = render_dockerfile(PythonToolchain(), _repo(tmp_path))
    assert text.startswith("FROM python:3.12-slim\n")
    assert "libfaketime" in text and "/usr/local/lib/libfaketime.so.1" in text
    assert "COPY . /repo" in text and "WORKDIR /repo" in text
    assert 'RUN pip install --no-cache-dir -e .' in text
    assert "SETUPTOOLS_SCM_PRETEND_VERSION" in text


def test_a_clean_build_keeps_tests_that_pass_both_runs(tmp_path):
    docker = FakeDocker()
    sandbox = FakeSandbox([
        _junit({"a": True, "b": True, "flaky": True, "broken": False}),
        _junit({"a": True, "b": True, "flaky": False, "broken": False}),
    ])
    result, pulled = _build(tmp_path, docker, sandbox)

    assert result.status == "ready" and result.reason == ""
    assert result.stable_tests == ["t::a", "t::b"]
    assert result.image_tag == repo_image_tag("acme_widgets") == "forge-repo-acme-widgets:latest"
    assert result.toolchain == "python"
    assert result.test_command == "python -m pytest -q -p no:cacheprovider --junitxml=/tmp/forge-report.xml"
    assert pulled == ["python:3.12-slim"]
    assert [image for image, _ in sandbox.commands] == [result.image_tag, result.image_tag]


def test_the_build_uses_its_own_dockerfile_and_ignores_the_repos(tmp_path):
    docker = FakeDocker()
    _build(tmp_path, docker, FakeSandbox([_junit({"a": True, "b": True})] * 2))
    (build_args,) = [args for args in docker.runs if args[0] == "build"]
    dockerfile = build_args[build_args.index("-f") + 1]
    assert dockerfile.endswith("Dockerfile.forge") and "/repo/" not in dockerfile
    assert (tmp_path / "Dockerfile.forge.dockerignore").read_text().strip() == ".git"
    assert (tmp_path / "repo" / "Dockerfile").read_text() == "FROM scratch\n"
    assert (tmp_path / "repo" / ".dockerignore").read_text() == "tests\n"


def test_no_toolchain_drops_the_repo_without_building(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "main.go").write_text("package main")
    docker = FakeDocker()
    result = build_repo_environment("x", root, "Go", docker=docker, sandbox=FakeSandbox([]),
                                    min_stable_tests=1, pull=lambda image: None)
    assert (result.status, result.reason) == ("dropped", "no supported toolchain")
    assert docker.runs == []


def test_an_install_failure_drops_the_repo_with_the_log(tmp_path):
    docker = FakeDocker(fail_build="ERROR: No matching distribution found for torch==0.0.1")
    result, _ = _build(tmp_path, docker, FakeSandbox([]))
    assert (result.status, result.reason) == ("dropped", "install failed")
    assert "torch==0.0.1" in result.log


def test_an_unreadable_report_drops_the_repo_and_removes_the_image(tmp_path):
    docker = FakeDocker()
    result, _ = _build(tmp_path, docker, FakeSandbox(["", ""]))
    assert (result.status, result.reason) == ("dropped", "baseline could not run")
    assert ("rmi", "-f", "forge-repo-acme-widgets:latest") in docker.discarded


def test_too_few_stable_tests_drops_the_repo(tmp_path):
    docker = FakeDocker()
    result, _ = _build(tmp_path, docker, FakeSandbox([_junit({"a": True, "b": False})] * 2), min_stable=2)
    assert (result.status, result.reason) == ("dropped", "1 stable tests, need 2")
    assert result.stable_tests == ["t::a"]
    assert ("rmi", "-f", "forge-repo-acme-widgets:latest") in docker.discarded


def test_the_baseline_runs_the_tests_from_the_repo_with_the_report_printed(tmp_path):
    sandbox = FakeSandbox([_junit({"a": True, "b": True})] * 2)
    _build(tmp_path, FakeDocker(), sandbox)
    command = sandbox.commands[0][1]
    assert command.startswith("cd /repo && ")
    assert command.endswith("cat /tmp/forge-report.xml")


def test_a_build_result_round_trips_through_json(tmp_path):
    result, _ = _build(tmp_path, FakeDocker(), FakeSandbox([_junit({"a": True, "b": True})] * 2))
    assert json.loads(json.dumps(result.to_dict()))["stable_tests"] == ["t::a", "t::b"]
