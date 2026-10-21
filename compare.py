"""Judge the pre-registered tests 1, 2 and 5 from runs/eval/*.npz (see PLAN.md).

    python compare.py        # prints verdicts, writes runs/compare.json; skips tests whose runs are missing
    python compare.py --suffix _v2   # same tests on the v2 runs (PLAN.md), writes runs/compare_v2.json

Bootstrap: 1000 image resamples with a fixed seed. A and T lists are aligned image by image and
every run of a split uses the same resample indices, so differences between runs are paired.
Seen vs unseen boards are different images, so those two subsets are resampled independently.
Resampled mAPs are cached in runs/eval/boot/.
"""
import argparse
import json
from pathlib import Path

import numpy as np

import evaluate as E

EVAL = E.ROOT / "runs" / "eval"
BOOT = EVAL / "boot"
N_BOOT = 1000
SUF = ""  # run-name suffix: "" = v1, "_v2" = v2 (set by --suffix)


def idx_for(split_n, seed):
    """Fixed resample indices for a list of split_n images."""
    r = np.random.default_rng(seed)
    return r.integers(0, split_n, (N_BOOT, split_n))


SUBS = {"test": None, "test_seen": "test_seen", "test_unseen": "test_unseen"}


def boot(run, sub):
    """(point mAP50-95, resampled mAP50-95[N_BOOT]) for one run on test or one of its subsets."""
    BOOT.mkdir(parents=True, exist_ok=True)
    cache = BOOT / f"{run}__{sub}.npy"
    m = E.Matches.load(EVAL / f"{run}__test.npz")
    base = np.arange(len(m.per_image)) if SUBS[sub] is None else np.array(E.subset_index(run[0], "test", sub))
    point = m.map(base)[0]
    if cache.exists():
        return point, np.load(cache)
    seed = {"test": 1, "test_seen": 2, "test_unseen": 3}[sub]
    vals = np.array([m.map(base[i])[0] for i in idx_for(len(base), seed)])
    np.save(cache, vals)
    return point, vals


def have(*runs):
    """True if every run has a test .npz."""
    return all((EVAL / f"{r}__test.npz").exists() for r in runs)


def ci(x):
    """95% percentile interval, in points."""
    return [100 * float(np.percentile(x, 2.5)), 100 * float(np.percentile(x, 97.5))]


def pts(x):
    """Fraction -> points, rounded for printing."""
    return round(100 * float(x), 2)


def test1(out):
    """Unseen boards are harder for arm A (mean over seeds): seen - unseen > 0, CI excludes 0."""
    runs = [f"A900_s0{SUF}", f"A900_s1{SUF}"]
    if not have(*runs):
        return
    seen = [boot(r, "test_seen") for r in runs]
    unseen = [boot(r, "test_unseen") for r in runs]
    d_point = np.mean([s[0] for s in seen]) - np.mean([u[0] for u in unseen])
    d_boot = np.mean([s[1] for s in seen], 0) - np.mean([u[1] for u in unseen], 0)
    lo, hi = ci(d_boot)
    verdict = "PASS" if lo > 0 else "FAIL"
    out["test1"] = {"verdict": verdict, "seen_minus_unseen": pts(d_point), "ci95": [lo, hi],
                    "seen": [pts(s[0]) for s in seen], "unseen": [pts(u[0]) for u in unseen]}
    print(f"TEST 1 {verdict}: A seen - unseen {pts(d_point):+.2f} [{lo:+.2f}, {hi:+.2f}] "
          f"(seen {out['test1']['seen']}, unseen {out['test1']['unseen']})")
    print("       caveat: unseen boards are also different boards, not only 'not trained on'.")


def pair(arm_a, arm_b, n, seeds, sub):
    """Mean-over-seeds difference b - a (paired), its CI, and the seed spread of each arm."""
    pa = [boot(f"{arm_a}{n}_s{s}{SUF}", sub) for s in seeds]
    pb = [boot(f"{arm_b}{n}_s{s}{SUF}", sub) for s in seeds]
    d = np.mean([b[0] for b in pb]) - np.mean([a[0] for a in pa])
    dboot = np.mean([b[1] for b in pb], 0) - np.mean([a[1] for a in pa], 0)
    spread = max(abs(pa[0][0] - pa[1][0]), abs(pb[0][0] - pb[1][0]))
    return {"diff": pts(d), "ci95": ci(dboot), "seed_spread": pts(spread),
            arm_a: [pts(a[0]) for a in pa], arm_b: [pts(b[0]) for b in pb]}


def judge_beats(r):
    """PASS if the mean difference is positive and larger than the larger seed spread, else TIE."""
    return "PASS" if r["diff"] > r["seed_spread"] and r["diff"] > 0 else "TIE"


def test2(out):
    """T beats A on the full test by more than the seed spread; unseen boards descriptive."""
    if not have(*(f"{a}900_s{s}{SUF}" for a in "AT" for s in (0, 1))):
        return
    r = pair("A", "T", 900, (0, 1), "test")
    r["verdict"] = judge_beats(r)
    r["unseen"] = pair("A", "T", 900, (0, 1), "test_unseen")
    r["seen"] = pair("A", "T", 900, (0, 1), "test_seen")
    out["test2"] = r
    print(f"TEST 2 {r['verdict']}: T - A {r['diff']:+.2f} [{r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f}] "
          f"vs seed spread {r['seed_spread']:.2f} (A {r['A']}, T {r['T']})")
    for k in ("seen", "unseen"):
        u = r[k]
        print(f"       {k:6s} (descriptive): T - A {u['diff']:+.2f} [{u['ci95'][0]:+.2f}, {u['ci95'][1]:+.2f}]"
              f" (A {u['A']}, T {u['T']})")


def test5(out):
    """Data curve: smallest size reaching 95% of the arm's own 900 mAP; T100 vs A100 judged."""
    sizes = [25, 50, 100, 225, 450, 900]
    curve = {}
    for arm in ("A", "T"):
        pts_ = {}
        for n in sizes:
            runs = [f"{arm}{n}_s{s}{SUF}" for s in (0, 1) if have(f"{arm}{n}_s{s}{SUF}")]
            if runs:
                pts_[n] = float(np.mean([boot(r, "test")[0] for r in runs]))
        curve[arm] = pts_
    if not all(len(curve[a]) == len(sizes) for a in curve):
        return
    r = {"curve": {a: {n: pts(v) for n, v in c.items()} for a, c in curve.items()}}
    for arm, c in curve.items():
        r[f"{arm}_95pct_at"] = next(n for n in sizes if c[n] >= 0.95 * c[900])
    j = pair("A", "T", 100, (0, 1), "test")
    j["verdict"] = judge_beats(j)
    r["T100_vs_A100"] = j
    out["test5"] = r
    print(f"TEST 5 {j['verdict']}: T100 - A100 {j['diff']:+.2f} vs seed spread {j['seed_spread']:.2f}; "
          f"95% of own 900 reached at A {r['A_95pct_at']}, T {r['T_95pct_at']}")
    for arm in ("A", "T"):
        print(f"       {arm}: " + "  ".join(f"{n}:{v:.2f}" for n, v in r["curve"][arm].items()))


def main():
    """Judge whatever tests have their runs, write runs/compare<suffix>.json."""
    global SUF
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="", help='run-name suffix, e.g. "_v2"')
    SUF = ap.parse_args().suffix
    out = {}
    for t in (test1, test2, test5):
        t(out)
    (E.ROOT / "runs" / f"compare{SUF}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
