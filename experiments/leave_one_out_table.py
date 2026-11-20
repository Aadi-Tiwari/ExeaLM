"""Tables of the leave-one-out ablation (issue #59) from results/leave_one_out/*.json.

    python experiments/leave_one_out_table.py            # print the tables
    python experiments/leave_one_out_table.py --write    # also fill results/leave_one_out/README.in.md into README.md

The score of a run is the mean of its four held-out bits per byte (English, Japanese wiki, Japanese web, code), as in
results/funding. A change against the control smaller than --noise (0.005 bits per byte) is called noise: it is inside
the seed-to-seed spread at this size.
"""
import argparse
import glob
import json
import os
import statistics

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "leave_one_out")
KINDS = ("en", "ja_wiki", "ja_web", "code")


def load():
    runs = {}
    for f in sorted(glob.glob(os.path.join(OUT, "*_seed*.json"))):
        r = json.load(open(f))
        runs.setdefault(r["config"], {})[r["seed"]] = r
    return runs


def score(r):
    return sum(r["bits_per_byte"][k] for k in KINDS) / len(KINDS)


def tables(noise):
    runs = load()
    if "control" not in runs:
        return "No control runs yet.\n"
    ctrl = {s: score(r) for s, r in runs["control"].items()}
    names = ["control"] + sorted((n for n in runs if n != "control"), key=lambda n: statistics.mean(map(score, runs[n].values())))
    seeds = sorted({s for n in runs for s in runs[n]})
    lines = ["| Left out | Seeds | " + " | ".join(f"Seed {s}" for s in seeds) + " | Mean | Std | Change against control | Paired change per seed | Reading |",
             "|---|---|" + "---|" * len(seeds) + "---|---|---|---|---|"]
    for n in names:
        sc = {s: score(r) for s, r in runs[n].items()}
        mean = statistics.mean(sc.values())
        sd = f"{statistics.stdev(sc.values()):.4f}" if len(sc) > 1 else "n/a"
        common = [s for s in sorted(sc) if s in ctrl]
        if n == "control" or not common:
            delta, paired, reading = "", "", ""
        else:
            d = [sc[s] - ctrl[s] for s in common]
            delta = f"{statistics.mean(sc[s] for s in common) - statistics.mean(ctrl[s] for s in common):+.4f}"
            paired = ", ".join(f"{x:+.4f}" for x in d)
            m = statistics.mean(d)
            reading = "noise" if abs(m) < noise else ("worse without it (type helps)" if m > 0 else "better without it (type hurts)")
            if reading != "noise" and len({x > 0 for x in d}) > 1:
                reading += ", seeds disagree"
        label = "nothing (control)" if n == "control" else n[len("leave_"):]
        cells = " | ".join(f"{sc[s]:.4f}" if s in sc else "" for s in seeds)
        lines.append(f"| {label} | {len(sc)} | {cells} | {mean:.4f} | {sd} | {delta} | {paired} | {reading} |")
    kinds = ["", "Bits per byte by kind of text, mean over seeds:", "",
             "| Left out | English | Japanese wiki | Japanese web | Code |", "|---|---|---|---|---|"]
    for n in names:
        label = "nothing (control)" if n == "control" else n[len("leave_"):]
        kinds.append(f"| {label} | " + " | ".join(f"{statistics.mean(r['bits_per_byte'][k] for r in runs[n].values()):.4f}" for k in KINDS) + " |")
    return "\n".join(lines + kinds) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise", type=float, default=0.005)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    text = tables(a.noise)
    print(text)
    if a.write:
        src = open(os.path.join(OUT, "README.in.md"), encoding="utf-8").read()
        open(os.path.join(OUT, "README.md"), "w", encoding="utf-8").write(src.replace("{{TABLES}}", text))


if __name__ == "__main__":
    main()
