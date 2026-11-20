"""Leave-one-out ablation of the hybrid's starting circuit menu at 13M (issue #59), as a resumable queue.

For each type T in D I GI M MI MIM EP G SM OS, the hybrid starts with T's weight set to 0 and trains 3,000 steps;
the control leaves nothing out. Every (configuration, seed) pair is one run of experiments/tokenizer_lm.py with the
arguments of the issue, and its result goes to results/leave_one_out/<config>_seed<k>.json. A pair whose file exists is
skipped, so running the same command again resumes after a crash, a reboot or the stop time.

    python experiments/leave_one_out.py                      # the whole queue, one run at a time
    python experiments/leave_one_out.py --dry_run            # print the queue and what is done
    python experiments/leave_one_out.py --seeds 0 --only control,leave_D
    python experiments/leave_one_out.py --stop_at 06:30 --min_free_gb 1.5 --lock path/to/.heavy.lock

--stop_at HH:MM starts no run that would end after that local time. --min_free_gb waits until that much memory would
stay free during a run, and ends a run of our own when free memory stays below it for 30 seconds. --lock takes a lock file around
each run (waiting while another holder's file is under 3 hours old).
"""
import argparse
import json
import os
import runpy
import statistics
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TYPES = ("D", "I", "GI", "M", "MI", "MIM", "EP", "G", "SM", "OS")
HYBRID = {"attn_layers": [1, 2, 4, 5], "init_weights": {"C": 4, "G": 0.25, "D": 0.25, "RP": 0}, "attn_gate": True, "qk_norm": True}
OUT = os.path.join(ROOT, "results", "leave_one_out")
SCRATCH = os.path.join(ROOT, "logs", "leave_one_out")


def config(name):
    cfg = json.loads(json.dumps(HYBRID))
    if name != "control":
        cfg["init_weights"][name[len("leave_"):]] = 0
    return cfg


def tag(name, seed):
    return f"f13_{name}_s{seed}"


def result_path(name, seed):
    return os.path.join(OUT, f"{name}_seed{seed}.json")


def free_gb():
    if sys.platform == "win32":
        out = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory"],
                             capture_output=True, text=True).stdout.strip()
        return float(out) / 1024 ** 2
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) / 1024 ** 2
    return float("inf")


def peak_rss_mb():
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]
        c = Counters()
        c.cb = ctypes.sizeof(c)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        ctypes.windll.psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(c), c.cb)
        return round(c.PeakWorkingSetSize / 2 ** 20)
    import resource
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)


def run_one(name, seed):
    """Child process: one run through tokenizer_lm.py's own argument parser, then the measurements added to its row."""
    import torch
    os.makedirs(SCRATCH, exist_ok=True)
    raw = os.path.join(SCRATCH, "raw")
    sys.argv = ["tokenizer_lm.py", "--toks", "bpe32k", "--models", "tonn_core", "--cfg", json.dumps(config(name)),
                "--seeds", str(seed), "--tag", tag(name, seed), "--out", raw]
    t0 = time.time()
    runpy.run_path(os.path.join(ROOT, "experiments", "tokenizer_lm.py"), run_name="__main__")
    wall = time.time() - t0
    row = json.load(open(os.path.join(raw, f"lm_compare_{tag(name, seed)}.json")))["rows"][0]
    row.update(config=name, left_out=None if name == "control" else name[len("leave_"):], wall_seconds=round(wall, 1),
               peak_rss_mb=peak_rss_mb(), torch=torch.__version__)
    if torch.cuda.is_available():
        row.update(device=torch.cuda.get_device_name(0), peak_vram_allocated_mb=round(torch.cuda.max_memory_allocated() / 2 ** 20),
                   peak_vram_reserved_mb=round(torch.cuda.max_memory_reserved() / 2 ** 20))
    os.makedirs(OUT, exist_ok=True)
    tmp = result_path(name, seed) + ".tmp"
    json.dump(row, open(tmp, "w"), indent=1)
    os.replace(tmp, result_path(name, seed))


class Lock:
    def __init__(self, path, who):
        self.path, self.who, self.held = path, who, False

    def acquire(self, log):
        if not self.path:
            return
        while True:
            try:
                if os.path.exists(self.path) and time.time() - os.path.getmtime(self.path) > 3 * 3600:
                    os.remove(self.path)
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, f"{self.who} {time.strftime('%Y-%m-%d %H:%M:%S')}".encode())
                os.close(fd)
                self.held = True
                return
            except (FileExistsError, PermissionError, FileNotFoundError):
                log(f"lock held by {self.read()}; waiting")
                time.sleep(60)

    def read(self):
        try:
            return open(self.path).read().strip()
        except OSError:
            return "?"

    def release(self):
        if self.held and os.path.exists(self.path) and self.read().startswith(self.who):
            os.remove(self.path)
        self.held = False


def seconds_until(hhmm):
    h, m = map(int, hhmm.split(":"))
    now = time.localtime()
    target = time.mktime(time.struct_time((now.tm_year, now.tm_mon, now.tm_mday, h, m, 0, 0, 0, -1)))
    return target - time.time() if target > time.time() else target + 86400 - time.time()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--only", default="", help="comma-separated configuration names (control, leave_D, ...)")
    ap.add_argument("--stop_at", default="", help="HH:MM local time: start no run that would end after it")
    ap.add_argument("--min_free_gb", type=float, default=0.0)
    ap.add_argument("--rss_gb", type=float, default=1.5, help="memory a run is expected to need, until one has been measured")
    ap.add_argument("--lock", default="")
    ap.add_argument("--who", default="leave-one-out-runner")
    ap.add_argument("--retries", type=int, default=6)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--one", nargs=2, metavar=("CONFIG", "SEED"), help="internal: run a single pair in this process")
    a = ap.parse_args()
    if a.one:
        return run_one(a.one[0], int(a.one[1]))

    names = ["control"] + [f"leave_{t}" for t in TYPES]
    if a.only:
        names = [n for n in a.only.split(",")]
    queue = [(n, int(s)) for s in a.seeds.split(",") for n in names]
    todo = [(n, s) for n, s in queue if not os.path.exists(result_path(n, s))]
    if a.dry_run:
        for n, s in queue:
            print(("done  " if (n, s) not in todo else "todo  ") + os.path.relpath(result_path(n, s), ROOT))
        print(f"{len(todo)} of {len(queue)} to run")
        return

    os.makedirs(SCRATCH, exist_ok=True)
    logf = open(os.path.join(ROOT, "logs", "leave_one_out.log"), "a", buffering=1)

    def log(msg):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(line, flush=True)
        logf.write(line + "\n")

    lock = Lock(a.lock, a.who)
    durations, rss_gb = [], a.rss_gb
    log(f"queue: {len(todo)} of {len(queue)} to run")
    for n, s in todo:
        for attempt in range(a.retries):
            est = statistics.median(durations) if durations else 0
            if a.stop_at and seconds_until(a.stop_at) < est:
                log(f"stop time {a.stop_at} reached; {len([1 for p in todo if not os.path.exists(result_path(*p))])} runs left")
                return
            while a.min_free_gb and free_gb() - rss_gb < a.min_free_gb:
                log(f"free memory {free_gb():.2f} GB too low for a run needing about {rss_gb:.1f} GB; waiting")
                time.sleep(60)
            lock.acquire(log)
            try:
                log(f"start {n} seed {s} (attempt {attempt + 1})")
                t0 = time.time()
                with open(os.path.join(SCRATCH, f"{n}_seed{s}.log"), "w") as out:
                    p = subprocess.Popen([sys.executable, os.path.abspath(__file__), "--one", n, str(s)], stdout=out,
                                         stderr=subprocess.STDOUT, cwd=ROOT)
                    low, strikes = False, 0
                    while p.poll() is None:
                        time.sleep(10)
                        strikes = strikes + 1 if a.min_free_gb and free_gb() < a.min_free_gb else 0
                        if strikes >= 3:
                            log(f"free memory under {a.min_free_gb} GB for 30 s: ending {n} seed {s} to give it back")
                            p.kill()
                            p.wait()
                            low = True
            finally:
                lock.release()
            mins = (time.time() - t0) / 60
            if p.returncode == 0 and os.path.exists(result_path(n, s)):
                row = json.load(open(result_path(n, s)))
                durations.append(time.time() - t0)
                rss_gb = max(rss_gb, row["peak_rss_mb"] / 1024)
                log(f"done {n} seed {s} in {mins:.1f} min: mean bpb {sum(row['bits_per_byte'].values()) / 4:.4f} {row['bits_per_byte']}")
                break
            log(f"FAILED {n} seed {s} after {mins:.1f} min (exit {p.returncode}{', low memory' if low else ''}); see logs/leave_one_out/{n}_seed{s}.log")
            time.sleep(60)
    log("queue finished")


if __name__ == "__main__":
    main()
