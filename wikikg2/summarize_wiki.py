"""Mean +/- unbiased std (ddof=1) over the seeded finals of ogbl-wikikg2, from the receipts in results/:
  rowA/final_s{s}.log          row A   single model: [valid]/[test] lines of train_wiki.py
  rowB/blend_s{s}.json         row B   A + 8 retrieval members, weights frozen on full valid
  rowBplus/blendaug_s{s}.json  row B+  + self-augmented members
  rowF/learned_s{s}.json       row F   A + 10 members + learned per-relation combiner
  rowCF/learned_dist_s{s}.json row C-F T=2 distilled student + members + combiner   [submitted, single model]
  rowCF/distill_s{s}.log               the student alone
  ensemble/learned_ens.json    ten-seed ensemble + members + combiner (one read)     [submitted, ensemble]
  ensemble/learned_loo{s}.json leave-one-out ensembles (9 of 10 seeds), ten reads -> its mean/std
  xfit/*.log                   cross-fitted (held-out) validation of the combiner rows, where available
    python summarize_wiki.py [--results results]
"""
import argparse, glob, json, os, re
import numpy as np


def stat(xs):
    xs = np.asarray(xs, float)
    if len(xs) == 0:
        return "n/a"
    sd = xs.std(ddof=1) if len(xs) > 1 else 0.0
    return f"{xs.mean():.4f} +/- {sd:.4f} (n={len(xs)})"


def logs(pattern, key):
    pat = re.compile(r"\[(valid|test)\] MRR ([0-9.]+)\s+hits@1 ([0-9.]+)\s+hits@3 ([0-9.]+)\s+hits@10 ([0-9.]+)")
    out = {"valid": [], "test": [], "h1": [], "h10": []}
    for f in sorted(glob.glob(pattern)):
        seen = {}
        for line in open(f):
            m = pat.search(line)
            if m:
                seen[m.group(1)] = tuple(float(x) for x in m.groups()[1:])
        if "test" in seen:
            out["test"].append(seen["test"][0]); out["h1"].append(seen["test"][1]); out["h10"].append(seen["test"][3])
        if "valid" in seen:
            out["valid"].append(seen["valid"][0])
    return out


def jsons(pattern):
    v, t = [], []
    for f in sorted(glob.glob(pattern)):
        d = json.load(open(f)); v.append(d["valid_in_sample"]); t.append(d["test_mrr"])
    return v, t


def main():
    p = argparse.ArgumentParser(); p.add_argument("--results", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    R = p.parse_args().results
    A = logs(f"{R}/rowA/final_s*.log", "test")
    print("row A   single k=8 model (329M)")
    print(f"  test MRR   {stat(A['test'])}   valid {stat(A['valid'])}   test hits@1 {np.mean(A['h1']):.4f} hits@10 {np.mean(A['h10']):.4f}")
    for name, pat, label in (("row B  ", f"{R}/rowB/blend_s*.json", "A + 8 retrieval members, frozen blend"),
                             ("row B+ ", f"{R}/rowBplus/blendaug_s*.json", "+ self-augmented members"),
                             ("row F  ", f"{R}/rowF/learned_s?.json", "A + 10 members + learned combiner"),
                             ("row C-F", f"{R}/rowCF/learned_dist_s*.json", "T=2 student + members + learned combiner  [submitted]")):
        v, t = jsons(pat)
        if t:
            print(f"{name} {label}\n  test MRR   {stat(t)}   valid (in-sample) {stat(v)}")
    S = logs(f"{R}/rowCF/distill_s*.log", "test")
    if S["test"]:
        print(f"  student alone: test {stat(S['test'])}   valid {stat(S['valid'])}")
    v, t = jsons(f"{R}/ensemble/learned_loo*.json")
    e = json.load(open(f"{R}/ensemble/learned_ens.json")) if os.path.exists(f"{R}/ensemble/learned_ens.json") else None
    print("ensemble  ten-seed ensemble + members + learned combiner  [submitted]")
    if e:
        print(f"  full ten-seed ensemble: test {e['test_mrr']:.4f}   valid (in-sample) {e['valid_in_sample']:.4f}  (one read)")
    if t:
        print(f"  leave-one-out (9 of 10 seeds), ten reads: test {stat(t)}   valid (in-sample) {stat(v)}")
    for f in sorted(glob.glob(f"{R}/xfit/*.log")):
        m = re.search(r"^\s*learned\s+([0-9.]+)", open(f).read(), re.M)
        if m:
            print(f"  cross-fitted valid, {os.path.basename(f)[:-4]}: {float(m.group(1)):.4f}")


if __name__ == "__main__":
    main()
