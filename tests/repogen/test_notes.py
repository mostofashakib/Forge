"""Task-writer notes for a repo environment carry everything the writer needs."""
from forge.repogen.notes import RepoMeta, repo_notes


def _meta(**overrides):
    fields = dict(
        full_name="acme/widgets", sha="0123456789abcdef" * 2 + "01234567", toolchain="python",
        test_command="python -m pytest -q -p no:cacheprovider --junitxml=/tmp/forge-report.xml",
        stable_tests=[f"tests.test_app::test_{i}" for i in range(40)],
    )
    fields.update(overrides)
    return RepoMeta(**fields)


def test_python_notes_name_the_repo_tests_and_grader_path():
    notes = repo_notes(_meta())
    assert "acme/widgets" in notes and "0123456789ab" in notes and "/repo" in notes
    assert "40 tests pass reliably" in notes
    assert "tests.test_app::test_0" in notes and "tests.test_app::test_39" not in notes
    assert "export PATH=/usr/local/bin:$PATH" in notes
    assert "python -m pytest -q tests/test_app.py::test_ok" in notes
    assert "--junitxml" not in notes
    assert "NotImplementedError" in notes
    # Class-based pytest ids keep the class as a node, not a folder.
    assert "tests.test_signer.TestSigner::test_sign is tests/test_signer.py::TestSigner::test_sign" in notes


def test_node_notes_use_jest_selection_and_js_stubs():
    notes = repo_notes(_meta(
        toolchain="node", test_command="npx jest --ci --json --outputFile=/tmp/forge-report.json",
        stable_tests=["src/math.test.ts::add sums"],
    ))
    assert "npx jest src/math.test.ts -t 'add sums'" in notes
    assert "throw new Error('not implemented')" in notes
    assert "1 tests pass reliably" in notes


def test_notes_forbid_editing_tests_and_require_failing_then_passing_checks():
    notes = repo_notes(_meta())
    assert "Never edit test files" in notes
    assert "fail after setup and pass after the golden steps" in notes
