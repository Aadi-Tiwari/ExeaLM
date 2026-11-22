"""Stage training data on disk: several files at once, resumable, and every file checked against the size and sha256
its host publishes.

    python -m exealm.data.stage --source ja/ja_wiki --plan_only             # files, sizes and total; downloads nothing
    python -m exealm.data.stage --source ja/ja_wiki --max_files 2 --out data/stage
    python -m exealm.data.stage --source IFM/TxT360-v2/txt360-qa --max_files 1 --workers 4
    python -m exealm.data.stage --mix v0 --max_files 1 --plan_only          # one file of every source in mix v0
    python -m exealm.data.stage --source en/en_wiki --max_tokens 0.5        # about 0.5B tokens of it
    python -m exealm.data.stage --source IFM/TxT360-v2 --max_gb 20          # at most 20 GB of it

A source is an LLM-jp Corpus v4 path (the keys of sources.LLMJP) or a Hugging Face dataset, "org/name" or
"org/name/folder". Both hosts keep the data files in Git LFS, which publishes each file's sha256 and size, so every
file is checked against them. A file is written to `<name>.part` and gets its final name only once its size and
sha256 match, so a file under its final name is complete. Stop at any point and run the same command again: finished
files are skipped, partial files continue where they stopped (from the start if the server ignores byte ranges), and
a file that fails its check is deleted so the next run fetches it fresh. `manifest.jsonl` in --out lists every
finished file with its source, url, size and sha256. Hugging Face files are pinned to the dataset's commit at
planning time.

--max_files, --max_gb and --max_tokens cap each source on its own. A capped source keeps files spread evenly through
its sorted list. --max_tokens needs a published token count (LLM-jp's, in sources.LLMJP) and turns it into bytes with
that source's bytes per token. LLM-jp counted with its own tokenizer, so the cap is an estimate.

LLM-jp files are already jsonl.gz with a `text` field. Hugging Face files stay as the host has them (parquet for K2);
`--to_jsonl DIR` turns the staged parquet files into the jsonl.gz files the rest of the data scripts read:

    python -m exealm.data.stage --out data/stage --to_jsonl data/raw
"""
import argparse
import gzip
import hashlib
import http.client
import json
import os
import shutil
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass

from .download import get_json, llmjp_files
from .sources import K2, LLMJP, LLMJP_RAW, MIX_V0

LLMJP_GRAPHQL = "https://gitlab.llm-jp.nii.ac.jp/api/graphql"
LLMJP_PROJECT = "datasets/llm-jp-corpus-v4"
DATA_SUFFIXES = (".jsonl.gz", ".jsonl", ".json.gz", ".jsonl.zst", ".parquet", ".arrow")
RETRY_HTTP = {408, 425, 429, 500, 502, 503, 504}


@dataclass(frozen=True)
class Item:
    source: str
    path: str           # where the file goes, relative to the output folder
    url: str
    size: int
    sha256: str


class CheckError(Exception):
    """A finished download did not match its published size or sha256."""


def _open(url, start=0, timeout=60):
    headers = {"Range": f"bytes={start}-"} if start else {}
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout)


def _sha256(path, block=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(block):
            h.update(b)
    return h


def _resumed(r, start):
    """Whether a response to a request for bytes start- really begins at start."""
    rng = r.headers.get("Content-Range", "")
    return getattr(r, "status", 200) == 206 and rng.startswith(f"bytes {start}-")


def fetch(item, out, retries=3, rehash=False, opener=_open, block=1 << 20, wait=2.0):
    """Download one item into the folder out. Returns "have" when the file is already there with the right size (with
    rehash, also the right sha256) and "got" after a download. Network errors and server errors are retried, each try
    continuing from the bytes already written. Raises CheckError when the downloaded file's size or sha256 is wrong;
    the partial file is deleted first."""
    dst = os.path.join(out, item.path)
    if os.path.exists(dst) and os.path.getsize(dst) == item.size:
        if not rehash or _sha256(dst).hexdigest() == item.sha256:
            return "have"
        os.remove(dst)
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    part = dst + ".part"
    have = os.path.getsize(part) if os.path.exists(part) else 0
    if have > item.size:
        os.remove(part)
        have = 0
    h = _sha256(part) if have else hashlib.sha256()
    for attempt in range(retries + 1):
        try:
            if have < item.size:
                with opener(item.url, have) as r:
                    if have and not _resumed(r, have):         # the server sent the whole file: start over
                        have, h = 0, hashlib.sha256()
                    with open(part, "ab" if have else "wb") as f:
                        while b := r.read(block):
                            f.write(b)
                            h.update(b)
                            have += len(b)
                if have < item.size:                           # read(n) ends quietly when the connection drops
                    raise http.client.IncompleteRead(b"", item.size - have)
            break
        except (OSError, http.client.HTTPException) as e:
            if isinstance(e, urllib.error.HTTPError) and e.code == 416:     # the server has fewer bytes than published
                os.remove(part)
                raise CheckError(f"{item.path}: the server has fewer than {have} bytes, expected {item.size}") from e
            if (isinstance(e, urllib.error.HTTPError) and e.code not in RETRY_HTTP) or attempt == retries:
                raise
            if os.path.exists(part) and os.path.getsize(part) != have:     # a write failed halfway: re-read it
                have = os.path.getsize(part)
                h = _sha256(part)
            time.sleep(min(wait * 2 ** attempt, 60))
    if have != item.size or h.hexdigest() != item.sha256:
        os.remove(part)
        raise CheckError(f"{item.path}: got {have} bytes with sha256 {h.hexdigest()}, "
                         f"expected {item.size} bytes with sha256 {item.sha256}")
    os.replace(part, dst)
    return "got"


def _missing_bytes(items, out):
    n = 0
    for it in items:
        dst = os.path.join(out, it.path)
        if os.path.exists(dst) and os.path.getsize(dst) == it.size:
            continue
        part = dst + ".part"
        n += it.size - (min(os.path.getsize(part), it.size) if os.path.exists(part) else 0)
    return n


def stage(items, out, workers=4, retries=3, rehash=False, opener=_open, log=print):
    """Fetch every item into out, `workers` files at a time, and add each finished file to out/manifest.jsonl. A failed
    file does not stop the others. Returns {"got": n, "have": n, "failed": [(path, error), ...]}."""
    paths = [it.path for it in items]
    if len(set(paths)) != len(paths):
        raise ValueError("two items have the same path")
    os.makedirs(out, exist_ok=True)
    need, free = _missing_bytes(items, out), shutil.disk_usage(out).free
    if need > free:
        raise SystemExit(f"{need / 1e9:.1f} GB still to download but only {free / 1e9:.1f} GB free in {out}")
    man = os.path.join(out, "manifest.jsonl")
    listed = set()
    if os.path.exists(man):
        with open(man, encoding="utf-8") as f:
            listed = {json.loads(line)["path"] for line in f if line.strip()}
    lock = threading.Lock()
    res = {"got": 0, "have": 0, "failed": []}

    def one(it):
        try:
            status = fetch(it, out, retries, rehash, opener)
        except Exception as e:                          # report it and let the other files finish
            with lock:
                res["failed"].append((it.path, f"{type(e).__name__}: {e}"))
            log(f"failed {it.path}: {e}")
            return
        with lock:
            res[status] += 1
            if it.path not in listed:
                with open(man, "a", encoding="utf-8") as f:
                    f.write(json.dumps(asdict(it), ensure_ascii=False) + "\n")
                listed.add(it.path)
        log(f"{status} {it.path} ({it.size / 1e6:.1f} MB)")

    with ThreadPoolExecutor(max(1, workers)) as ex:
        list(ex.map(one, items))
    return res


def _post_json(url, body):
    return get_json(urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"}))


def llmjp_items(subset, max_files=None, list_files=llmjp_files, post=_post_json, batch=50):
    """The files of one LLM-jp sub-corpus, with the size and sha256 of each from GitLab's GraphQL API (one request per
    `batch` files)."""
    files = spread([f for f in list_files(subset) if f.endswith(DATA_SUFFIXES)], max_files)
    items = []
    for i in range(0, len(files), batch):
        paths = files[i:i + batch]
        query = ('{ project(fullPath: %s) { repository { blobs(ref: "main", paths: %s) { nodes { path rawSize lfsOid } '
                 '} } } }' % (json.dumps(LLMJP_PROJECT), json.dumps(paths)))
        nodes = post(LLMJP_GRAPHQL, {"query": query})["data"]["project"]["repository"]["blobs"]["nodes"]
        info = {n["path"]: n for n in nodes}
        for p in paths:
            n = info.get(p)
            if not n or not n.get("lfsOid"):
                raise ValueError(f"no sha256 published for {p} (not in Git LFS?)")
            items.append(Item(subset, "llm-jp-v4/" + p, LLMJP_RAW + p, int(n["rawSize"]), n["lfsOid"]))
    return items


def hf_items(source, max_files=None, api=None):
    """The data files of a Hugging Face dataset, "org/name" or "org/name/folder", pinned to its current commit."""
    if api is None:
        from huggingface_hub import HfApi
        api = HfApi()
    parts = source.split("/")
    repo, folder = "/".join(parts[:2]), "/".join(parts[2:]) or None
    sha = api.dataset_info(repo).sha
    items = []
    for f in api.list_repo_tree(repo, path_in_repo=folder, repo_type="dataset", recursive=True, revision=sha):
        lfs = getattr(f, "lfs", None)
        if lfs and f.path.endswith(DATA_SUFFIXES):
            url = f"https://huggingface.co/datasets/{repo}/resolve/{sha}/{f.path}"
            items.append(Item(source, f"hf/{repo}/{f.path}", url, int(lfs.size), lfs.sha256))
    return spread(sorted(items, key=lambda it: it.path), max_files)


def spread(xs, k=None):
    """k of the xs, evenly spaced through the list (all of them when k is None or at least len(xs)), so a sample
    covers the whole source and not only the start of its sorted file list."""
    n = len(xs)
    if k is None or k >= n:
        return list(xs)
    return [xs[i * n // k] for i in range(k)]


def _coarse_to_fine(n):
    """Every index below n, in the order the first, the middle, the quarters, the eighths and so on reach them."""
    order, seen, k = [], set(), 1
    while len(order) < n:
        for i in spread(range(n), k):
            if i not in seen:
                seen.add(i)
                order.append(i)
        k *= 2
    return order


def cap(items, max_files=None, max_bytes=None):
    """At most max_files of the items and at most max_bytes in all (but at least one file), spread through the list.
    Under a byte cap, files are tried coarse to fine through the list and each one that still fits is kept, so a few
    large files do not crowd out the rest."""
    k = len(items) if max_files is None else min(max_files, len(items))
    if max_bytes is None or not items:
        return spread(items, k)
    order = _coarse_to_fine(len(items))
    keep, used = [], 0
    for i in order:
        if len(keep) == k:
            break
        if used + items[i].size <= max_bytes:
            keep.append(i)
            used += items[i].size
    return [items[i] for i in sorted(keep or order[:1])]


def plan(sources, max_files=None, max_gb=None, max_tokens=None, llmjp=llmjp_items, hf=hf_items, log=print):
    """Items for each source. Each cap applies to every source on its own: max_files files, max_gb gigabytes, or
    max_tokens billion tokens. A token cap needs a published token count (sources.LLMJP); it is turned into bytes with
    the source's own bytes per token. A file is never split, so a source whose smallest step is over its cap keeps one
    file, and the plan says so."""
    items = []
    for s in sources:
        get = llmjp if s in LLMJP else hf
        if max_gb is None and max_tokens is None:
            items += get(s, max_files)
            continue
        if max_tokens is not None and not LLMJP.get(s, (None,))[0]:
            raise SystemExit(f"no published token count for {s}; cap it with --max_gb instead")
        got = get(s, None)
        limits = [max_gb * 1e9] if max_gb is not None else []
        if max_tokens is not None:
            limits.append(max_tokens / LLMJP[s][0] * sum(it.size for it in got))
        kept = cap(got, max_files, min(limits))
        if sum(it.size for it in kept) > min(limits):
            log(f"note: {s} keeps one file of {kept[0].size / 1e9:.2f} GB, over its cap of {min(limits) / 1e9:.2f} GB")
        items += kept
    return items


def to_jsonl(staged, out, batch=1000, log=print):
    """Turn the staged Hugging Face parquet files listed in staged/manifest.jsonl into jsonl.gz files of {"text",
    "source"} in out: the format download.py writes and quality.py reads. K2 subsets go through their verified
    formatters (rows without text are dropped, and K2 reads only the text column); other datasets use
    download.extract_text. Finished files are skipped. Returns {output file: documents written} for this run."""
    import pyarrow.parquet as pq

    from .download import extract_text
    from .formatters import get_k2_formatter
    with open(os.path.join(staged, "manifest.jsonl"), encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]
    os.makedirs(out, exist_ok=True)
    written = {}
    for e in entries:
        parts = e["path"].split("/")
        if parts[0] != "hf" or not e["path"].endswith(".parquet"):
            continue
        repo, subset = "/".join(parts[1:3]), (parts[3] if len(parts) > 4 else None)
        k2 = repo in K2
        fmt = get_k2_formatter(repo, subset) if k2 else extract_text
        dst = os.path.join(out, "hf__" + "__".join(parts[1:])[:-len(".parquet")] + ".jsonl.gz")
        if os.path.exists(dst):
            log(f"have {dst}")
            continue
        source, n = (f"{repo}/{subset}" if subset else repo), 0
        rows = pq.ParquetFile(os.path.join(staged, *parts)).iter_batches(batch, columns=["text"] if k2 else None)
        with gzip.open(dst + ".part", "wt", encoding="utf-8") as fo:
            for b in rows:
                for row in b.to_pylist():
                    text = fmt(row)
                    if text:
                        fo.write(json.dumps({"text": text, "source": source}, ensure_ascii=False) + "\n")
                        n += 1
        os.replace(dst + ".part", dst)
        written[dst] = n
        log(f"wrote {n} documents to {dst}")
    return written


def mix_sources(name):
    if name == "v0":
        return list(MIX_V0)
    raise ValueError(f"unknown mix {name}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Download data files, resumable and checked against their sha256.")
    ap.add_argument("--source", action="append", default=[],
                    help="an LLM-jp path or a Hugging Face dataset (org/name or org/name/folder); can repeat")
    ap.add_argument("--mix", choices=["v0"], help="every source of a mix in sources.py (mix v0 names its sources)")
    ap.add_argument("--max_files", type=int, default=None, help="at most this many files per source (a sample)")
    ap.add_argument("--max_gb", type=float, default=None, help="at most this many GB per source")
    ap.add_argument("--max_tokens", type=float, default=None,
                    help="at most about this many billion tokens per source (LLM-jp: from its published counts)")
    ap.add_argument("--out", default="data/stage")
    ap.add_argument("--workers", type=int, default=4, help="files downloaded at once")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--rehash", action="store_true", help="re-check the sha256 of finished files, fetch failures again")
    ap.add_argument("--plan_only", action="store_true", help="print the files and sizes, download nothing")
    ap.add_argument("--to_jsonl", default=None, metavar="DIR",
                    help="instead of downloading, turn the parquet files staged in --out into jsonl.gz files in DIR")
    a = ap.parse_args(argv)
    if a.to_jsonl:
        to_jsonl(a.out, a.to_jsonl)
        return
    sources = a.source + (mix_sources(a.mix) if a.mix else [])
    if not sources:
        ap.error("give --source or --mix")
    unknown = [s for s in sources if s not in LLMJP and s.split("/")[0] in ("ja", "en", "ko", "zh", "code")]
    if unknown:
        ap.error(f"not an LLM-jp path in sources.LLMJP: {', '.join(unknown)}")
    items = plan(sources, a.max_files, a.max_gb, a.max_tokens)
    for s in sources:
        mine = [it for it in items if it.source == s]
        print(f"{s:36s} {len(mine):6d} files {sum(it.size for it in mine) / 1e9:10.2f} GB")
    print(f"{'total':36s} {len(items):6d} files {sum(it.size for it in items) / 1e9:10.2f} GB")
    if a.plan_only:
        return
    res = stage(items, a.out, a.workers, a.retries, a.rehash)
    print(f"got {res['got']}, already had {res['have']}, failed {len(res['failed'])}")
    for path, err in res["failed"]:
        print(f"  {path}: {err}")
    if res["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
