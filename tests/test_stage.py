import hashlib
import json
import os
import random
import threading
from collections import namedtuple
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from exealm.data import stage as st
from exealm.data.stage import CheckError, Item, fetch, hf_items, llmjp_items, plan


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        srv = self.server
        data = srv.files.get(self.path)
        rng = self.headers.get("Range")
        srv.requests.append((self.path, rng))
        if data is None:
            self.send_error(404)
            return
        if self.path in srv.fail:
            srv.fail.discard(self.path)
            self.send_error(503)
            return
        start = int(rng.split("=")[1].split("-")[0]) if rng and srv.ranges else 0
        if start and start >= len(data):
            self.send_error(416)
            return
        body = data[start:]
        self.send_response(206 if start else 200)
        if start:
            self.send_header("Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        cut = srv.cut.pop(self.path, None)
        self.wfile.write(body if cut is None else body[:cut])

    def log_message(self, *args):
        pass


@pytest.fixture
def server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    srv.files, srv.requests, srv.cut, srv.fail, srv.ranges = {}, [], {}, set(), True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv
    srv.shutdown()
    srv.server_close()


def add(srv, name, data):
    """Serve data at /name and return the matching item."""
    srv.files["/" + name] = data
    return Item("test", "sub/" + name, f"http://127.0.0.1:{srv.server_port}/{name}", len(data),
                hashlib.sha256(data).hexdigest())


def blob(n, seed=0):
    return random.Random(seed).randbytes(n)


def read(path):
    with open(path, "rb") as f:
        return f.read()


def test_download_is_checked_and_listed_once(server, tmp_path):
    data = blob(300_000)
    it = add(server, "a.jsonl.gz", data)
    res = st.stage([it], str(tmp_path), workers=1, log=lambda s: None)
    assert res == {"got": 1, "have": 0, "failed": []}
    assert read(tmp_path / "sub" / "a.jsonl.gz") == data
    assert not os.path.exists(tmp_path / "sub" / "a.jsonl.gz.part")
    res = st.stage([it], str(tmp_path), workers=1, log=lambda s: None)
    assert res == {"got": 0, "have": 1, "failed": []}
    assert len(server.requests) == 1                     # the second run downloaded nothing
    lines = (tmp_path / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in lines] == [{"source": "test", "path": "sub/a.jsonl.gz", "url": it.url,
                                                     "size": len(data), "sha256": it.sha256}]


def test_a_cut_download_continues_from_the_bytes_it_has(server, tmp_path):
    data = blob(300_000, 1)
    it = add(server, "b.parquet", data)
    server.cut["/b.parquet"] = 100_000
    with pytest.raises(Exception):
        fetch(it, str(tmp_path), retries=0, block=4096)
    assert os.path.getsize(tmp_path / "sub" / "b.parquet.part") == 100_000
    assert fetch(it, str(tmp_path), retries=0, block=4096) == "got"
    assert read(tmp_path / "sub" / "b.parquet") == data
    assert server.requests == [("/b.parquet", None), ("/b.parquet", "bytes=100000-")]


def test_retries_inside_one_call_resume_too(server, tmp_path):
    data = blob(200_000, 2)
    it = add(server, "c.jsonl", data)
    server.cut["/c.jsonl"] = 50_000
    assert fetch(it, str(tmp_path), retries=1, block=4096, wait=0) == "got"
    assert read(tmp_path / "sub" / "c.jsonl") == data
    assert server.requests[-1] == ("/c.jsonl", "bytes=50000-")


def test_server_errors_are_retried(server, tmp_path):
    data = blob(10_000, 3)
    it = add(server, "d.jsonl", data)
    server.fail.add("/d.jsonl")
    assert fetch(it, str(tmp_path), retries=1, wait=0) == "got"
    assert read(tmp_path / "sub" / "d.jsonl") == data


def test_a_shorter_file_on_the_server_fails_the_check(server, tmp_path):
    data = blob(30_000, 11)
    it = add(server, "s.jsonl", data)
    server.files["/s.jsonl"] = data[:10_000]
    server.ranges = True
    with pytest.raises(Exception):
        fetch(it, str(tmp_path), retries=0)
    assert os.path.getsize(tmp_path / "sub" / "s.jsonl.part") == 10_000      # kept: it looks like a cut connection
    with pytest.raises(CheckError):
        fetch(it, str(tmp_path), retries=0)                                # the next request runs past the end
    assert not os.path.exists(tmp_path / "sub" / "s.jsonl.part")


def test_missing_file_is_not_retried(server, tmp_path):
    it = Item("test", "sub/none.jsonl", f"http://127.0.0.1:{server.server_port}/none.jsonl", 10, "0" * 64)
    with pytest.raises(OSError):
        fetch(it, str(tmp_path), retries=3, wait=0)
    assert len(server.requests) == 1


def test_a_server_that_ignores_ranges_gets_a_fresh_start(server, tmp_path):
    data = blob(120_000, 4)
    it = add(server, "e.jsonl.gz", data)
    server.ranges = False
    os.makedirs(tmp_path / "sub")
    (tmp_path / "sub" / "e.jsonl.gz.part").write_bytes(data[:40_000])
    assert fetch(it, str(tmp_path), retries=0) == "got"
    assert read(tmp_path / "sub" / "e.jsonl.gz") == data       # not the first 40,000 bytes twice
    assert server.requests == [("/e.jsonl.gz", "bytes=40000-")]


def test_wrong_bytes_fail_the_check_and_leave_nothing_behind(server, tmp_path):
    good, bad = add(server, "f.jsonl", blob(5_000, 5)), add(server, "g.jsonl", blob(5_000, 6))
    bad = replace(bad, sha256="0" * 64)
    with pytest.raises(CheckError):
        fetch(bad, str(tmp_path), retries=0)
    assert not os.path.exists(tmp_path / "sub" / "g.jsonl") and not os.path.exists(tmp_path / "sub" / "g.jsonl.part")
    res = st.stage([good, bad], str(tmp_path), workers=2, log=lambda s: None)
    assert res["got"] == 1 and [p for p, _ in res["failed"]] == ["sub/g.jsonl"]
    lines = (tmp_path / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["path"] for line in lines] == ["sub/f.jsonl"]


def test_a_stale_partial_file_is_caught_and_the_next_run_repairs_it(server, tmp_path):
    data = blob(50_000, 7)
    it = add(server, "h.jsonl", data)
    os.makedirs(tmp_path / "sub")
    (tmp_path / "sub" / "h.jsonl.part").write_bytes(b"x" * 20_000)       # left by another version of the file
    with pytest.raises(CheckError):
        fetch(it, str(tmp_path), retries=0)
    assert not os.path.exists(tmp_path / "sub" / "h.jsonl.part")
    assert fetch(it, str(tmp_path), retries=0) == "got"
    assert read(tmp_path / "sub" / "h.jsonl") == data


def test_rehash_fetches_a_damaged_file_again(server, tmp_path):
    data = blob(30_000, 8)
    it = add(server, "i.jsonl", data)
    os.makedirs(tmp_path / "sub")
    damaged = bytearray(data)
    damaged[100] ^= 1
    (tmp_path / "sub" / "i.jsonl").write_bytes(bytes(damaged))
    assert fetch(it, str(tmp_path)) == "have"                          # same size, so only --rehash can tell
    assert fetch(it, str(tmp_path), rehash=True) == "got"
    assert read(tmp_path / "sub" / "i.jsonl") == data


def test_many_files_at_once(server, tmp_path):
    items = [add(server, f"m{i}.jsonl", blob(20_000 + i, 10 + i)) for i in range(12)]
    res = st.stage(items, str(tmp_path), workers=4, log=lambda s: None)
    assert res == {"got": 12, "have": 0, "failed": []}
    for it in items:
        assert hashlib.sha256(read(tmp_path / it.path)).hexdigest() == it.sha256
    assert len((tmp_path / "manifest.jsonl").read_text(encoding="utf-8").splitlines()) == 12


def test_stage_refuses_when_the_disk_is_too_small(server, tmp_path, monkeypatch):
    it = add(server, "j.jsonl", blob(1_000, 9))
    monkeypatch.setattr(st.shutil, "disk_usage", lambda p: namedtuple("U", "total used free")(10, 10, 999))
    with pytest.raises(SystemExit, match="still to download"):
        st.stage([it], str(tmp_path), log=lambda s: None)
    assert server.requests == []


def test_same_path_twice_is_an_error(server, tmp_path):
    it = add(server, "k.jsonl", blob(10, 0))
    with pytest.raises(ValueError):
        st.stage([it, it], str(tmp_path))


def test_llmjp_plan_reads_sizes_and_sha256_in_batches():
    files = [f"ja/ja_wiki/{i:04d}.jsonl.gz" for i in range(7)] + ["ja/ja_wiki/README.md"]
    calls = []

    def post(url, body):
        paths = json.loads(body["query"].split("paths: ")[1].split(")")[0])
        calls.append(paths)
        nodes = [{"path": p, "rawSize": str(1000 + int(p[-13:-9])), "lfsOid": p[-13:-9] * 16} for p in paths]
        return {"data": {"project": {"repository": {"blobs": {"nodes": nodes}}}}}

    items = llmjp_items("ja/ja_wiki", max_files=5, list_files=lambda s: files, post=post, batch=2)
    assert [len(c) for c in calls] == [2, 2, 1]
    assert [it.size for it in items] == [1000, 1001, 1002, 1003, 1004]
    assert items[0] == Item("ja/ja_wiki", "llm-jp-v4/ja/ja_wiki/0000.jsonl.gz",
                            st.LLMJP_RAW + "ja/ja_wiki/0000.jsonl.gz", 1000, "0000" * 16)


def test_llmjp_plan_refuses_a_file_without_a_sha256():
    def post(url, body):
        return {"data": {"project": {"repository": {"blobs": {"nodes": [{"path": "ja/ja_wiki/0000.jsonl.gz",
                                                                          "rawSize": "134", "lfsOid": None}]}}}}}

    with pytest.raises(ValueError, match="no sha256"):
        llmjp_items("ja/ja_wiki", list_files=lambda s: ["ja/ja_wiki/0000.jsonl.gz"], post=post)


def test_hf_plan_keeps_lfs_data_files_pinned_to_one_commit():
    def f(path, size=None, sha=None):
        return SimpleNamespace(path=path, size=size, lfs=SimpleNamespace(size=size, sha256=sha) if sha else None)

    tree = [f("README.md", 3000), f("qa/b.parquet", 20, "b" * 64), f("qa/a.parquet", 10, "a" * 64),
            f("qa/notes.txt", 5, "c" * 64), SimpleNamespace(path="qa")]
    seen = {}

    class Api:
        def dataset_info(self, repo):
            return SimpleNamespace(sha="abc123")

        def list_repo_tree(self, repo, path_in_repo, repo_type, recursive, revision):
            seen.update(repo=repo, folder=path_in_repo, revision=revision)
            return tree

    items = hf_items("IFM/TxT360-v2/qa", api=Api())
    assert seen == {"repo": "IFM/TxT360-v2", "folder": "qa", "revision": "abc123"}
    assert items == [Item("IFM/TxT360-v2/qa", "hf/IFM/TxT360-v2/qa/a.parquet",
                          "https://huggingface.co/datasets/IFM/TxT360-v2/resolve/abc123/qa/a.parquet", 10, "a" * 64),
                     Item("IFM/TxT360-v2/qa", "hf/IFM/TxT360-v2/qa/b.parquet",
                          "https://huggingface.co/datasets/IFM/TxT360-v2/resolve/abc123/qa/b.parquet", 20, "b" * 64)]
    assert len(hf_items("IFM/TxT360-v2/qa", max_files=1, api=Api())) == 1


def test_plan_sends_each_source_to_its_host():
    got = plan(["ja/ja_wiki", "IFM/TxT360-v2"], max_files=3, llmjp=lambda s, n: [("llmjp", s, n)],
               hf=lambda s, n: [("hf", s, n)])
    assert got == [("llmjp", "ja/ja_wiki", 3), ("hf", "IFM/TxT360-v2", 3)]


def test_command_line_plan_only_prints_totals_and_downloads_nothing(monkeypatch, capsys):
    items = [Item("ja/ja_wiki", f"llm-jp-v4/ja/ja_wiki/{i}.jsonl.gz", "http://x", 2_000_000_000, "0" * 64)
             for i in range(3)]
    monkeypatch.setattr(st, "plan", lambda sources, n: items)
    monkeypatch.setattr(st, "stage", lambda *a, **k: pytest.fail("plan_only must not download"))
    st.main(["--source", "ja/ja_wiki", "--plan_only"])
    out = capsys.readouterr().out.splitlines()
    assert out[-1].split() == ["total", "3", "files", "6.00", "GB"]


def test_command_line_refuses_a_mistyped_llmjp_path(monkeypatch):
    monkeypatch.setattr(st, "plan", lambda sources, n: pytest.fail("no planning for a bad path"))
    with pytest.raises(SystemExit):
        st.main(["--source", "ja/ja_wikii", "--plan_only"])
