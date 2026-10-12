"""Snapshots: bounded download, safe extraction, file selection, and safety screening."""
import io
import tarfile

import httpx
import pytest

from forge.repogen.snapshot import SnapshotError, fetch_tarball, safe_extract, screen_repo, select_files
from forge.safety.pseudonymizer import Pseudonymizer
from forge.safety.service import DatasetSafetyService, PiiScrubbing


def _tarball(path, members):
    """members: (name, bytes | None, kind, linkname) with kind file, dir, symlink, hardlink, fifo, chr."""
    with tarfile.open(path, "w:gz") as tar:
        for name, data, kind, link in members:
            info = tarfile.TarInfo(name)
            if kind == "file":
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
                continue
            info.type = {"dir": tarfile.DIRTYPE, "symlink": tarfile.SYMTYPE, "hardlink": tarfile.LNKTYPE,
                         "fifo": tarfile.FIFOTYPE, "chr": tarfile.CHRTYPE}[kind]
            info.linkname = link or ""
            tar.addfile(info)
    return path


def test_extract_strips_the_top_folder(tmp_path):
    tarball = _tarball(tmp_path / "r.tgz", [
        ("acme-widgets-abc/", None, "dir", None),
        ("acme-widgets-abc/src/app.py", b"print(1)\n", "file", None),
        ("acme-widgets-abc/README.md", b"# hi\n", "file", None),
    ])
    root = safe_extract(tarball, tmp_path / "out", max_bytes=10_000)
    assert (root / "src" / "app.py").read_text() == "print(1)\n"
    assert (root / "README.md").exists()


@pytest.mark.parametrize("member", [
    ("top/../../escape.txt", b"x", "file", None),
    ("/etc/evil", b"x", "file", None),
    ("top/link", None, "symlink", "/etc/passwd"),
    ("top/link", None, "symlink", "../../outside"),
    ("top/dev", None, "chr", None),
    ("top/pipe", None, "fifo", None),
])
def test_hostile_members_reject_the_whole_archive(tmp_path, member):
    tarball = _tarball(tmp_path / "r.tgz", [("top/ok.py", b"x", "file", None), member])
    with pytest.raises(SnapshotError):
        safe_extract(tarball, tmp_path / "out", max_bytes=10_000)
    assert not (tmp_path / "escape.txt").exists()


def test_an_archive_that_expands_past_the_cap_is_rejected(tmp_path):
    tarball = _tarball(tmp_path / "r.tgz", [("top/big.txt", b"a" * 5000, "file", None)])
    with pytest.raises(SnapshotError, match="expands"):
        safe_extract(tarball, tmp_path / "out", max_bytes=1000)


def test_an_archive_with_several_top_folders_is_rejected(tmp_path):
    tarball = _tarball(tmp_path / "r.tgz", [("a/x.py", b"x", "file", None), ("b/y.py", b"y", "file", None)])
    with pytest.raises(SnapshotError, match="one top-level"):
        safe_extract(tarball, tmp_path / "out", max_bytes=10_000)


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_fetch_downloads_the_pinned_commit_with_the_token(tmp_path):
    seen = {}

    def handler(request):
        seen["url"], seen["auth"] = str(request.url), request.headers.get("authorization")
        return httpx.Response(200, content=b"tarball-bytes")

    path = fetch_tarball("acme/widgets", "a" * 40, "tok", tmp_path / "r.tgz", max_bytes=100, http=_client(handler))
    assert path.read_bytes() == b"tarball-bytes"
    assert seen == {"url": f"https://api.github.com/repos/acme/widgets/tarball/{'a' * 40}", "auth": "Bearer tok"}


def test_fetch_stops_at_the_size_cap_and_leaves_no_file(tmp_path):
    client = _client(lambda request: httpx.Response(200, content=b"x" * 500))
    with pytest.raises(SnapshotError, match="larger than"):
        fetch_tarball("acme/widgets", "a" * 40, "tok", tmp_path / "r.tgz", max_bytes=100, http=client)
    assert not (tmp_path / "r.tgz").exists()


def test_fetch_reports_http_errors(tmp_path):
    client = _client(lambda request: httpx.Response(404, text="Not Found"))
    with pytest.raises(SnapshotError, match="404"):
        fetch_tarball("acme/widgets", "a" * 40, "tok", tmp_path / "r.tgz", max_bytes=100, http=client)


def test_fetch_rejects_a_sha_that_is_not_a_commit_id(tmp_path):
    with pytest.raises(ValueError):
        fetch_tarball("acme/widgets", "main", "tok", tmp_path / "r.tgz", max_bytes=100, http=_client(None))


def _write(root, rel, data):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data if isinstance(data, bytes) else data.encode())


def test_select_files_skips_noise_and_keeps_source(tmp_path):
    for rel, data in {
        "src/app.py": "x = 1", "tests/test_app.py": "def test(): pass", "README.md": "# hi",
        "node_modules/lib/index.js": "x", ".git/config": "x", "dist/bundle.js": "x", "build/out.py": "x",
        "vendor/dep.py": "x", "static/app.min.js": "x", "package-lock.json": "{}", "poetry.lock": "x",
        "logo.png": b"\x89PNG\x00\x00", "big.txt": "a" * 1_100_000,
    }.items():
        _write(tmp_path, rel, data)
    selected = sorted(str(p) for p in select_files(tmp_path))
    assert selected == ["README.md", "src/app.py", "tests/test_app.py"]


class NoNames:
    def detect_many(self, texts):
        return [[] for _ in texts]


class NoAdult:
    def detect(self, texts):
        return []


def _service():
    return DatasetSafetyService(NoAdult(), PiiScrubbing(Pseudonymizer(b"k"), NoNames()))


def test_screening_rewrites_pii_in_place_and_accepts_the_repo(tmp_path):
    _write(tmp_path, "src/app.py", 'OWNER = "jane.doe@realcorp.com"\n')
    _write(tmp_path, "tests/test_app.py", 'assert OWNER == "jane.doe@realcorp.com"\n')
    verdict = screen_repo(tmp_path, select_files(tmp_path), _service())

    assert verdict.accepted and verdict.rejections == {}
    app, test = (tmp_path / "src/app.py").read_text(), (tmp_path / "tests/test_app.py").read_text()
    assert "realcorp" not in app + test
    # One value, one pseudonym, so the test still compares equal strings.
    assert app.split('"')[1] == test.split('"')[1]
    assert verdict.replacements == {"email": 2}


def test_one_flagged_file_rejects_the_repo_and_names_it(tmp_path):
    _write(tmp_path, "src/app.py", "x = 1\n")
    _write(tmp_path, "src/vendored.c", "/* SPDX-License-Identifier: GPL-2.0-only */\n")
    verdict = screen_repo(tmp_path, select_files(tmp_path), _service())
    assert not verdict.accepted
    assert verdict.rejections == {"src/vendored.c": ["copyright"]}


def test_undecodable_files_are_left_alone(tmp_path):
    _write(tmp_path, "data/latin1.txt", "caf\xe9".encode("latin-1"))
    verdict = screen_repo(tmp_path, select_files(tmp_path), _service())
    assert verdict.accepted
    assert (tmp_path / "data/latin1.txt").read_bytes() == "caf\xe9".encode("latin-1")
