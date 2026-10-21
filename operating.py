"""Operating points: pre-registered tests 3 and 4 from runs/eval/*.preds.npz (see PLAN.md).

    python operating.py              # prints verdicts, writes runs/operating.json
    python operating.py --iou 0.33   # POST-HOC sensitivity (decided after seeing results, see
                                     # PLAN.md 2026-10-20): writes runs/operating_iou033.json
    python operating.py --suffix _v2 # same on the v2 runs (PLAN.md): writes runs/operating_v2.json

A defect is caught when a detection at or above the threshold overlaps it with IoU >= 0.5.
Primary = class-agnostic (any class: the board gets pulled either way); the class-aware variant
is printed as a sensitivity. A false alarm is a detection that catches nothing. Matching is
greedy in score order, so each detection's matched/unmatched flag does not depend on the
threshold and one matching pass serves every threshold. Thresholds are chosen on val only.
"""
import argparse
import json

import numpy as np

import evaluate as E

EVAL = E.ROOT / "runs" / "eval"
IOU = 0.5
TARGET_RECALL = 0.99
COST_RATIOS = (1, 10, 50, 200)
N_BOOT = 1000


def load(run, split):
    """Raw detections of one run on one split, grouped per image."""
    z = np.load(EVAL / f"{run}__{split}.preds.npz")
    images = list(z["images"])
    per = [(z["cls"][z["img"] == i], z["xyxy"][z["img"] == i], z["score"][z["img"] == i])
           for i in range(len(images))]
    return images, per


def match(images, per, class_aware):
    """Per image: (scores[d], tp[d] bool, n_gt). Greedy by score, best-IoU unmatched gt."""
    out = []
    for img, (c, b, s) in zip(images, per):
        gc, gb = E.load_gt(img)
        order = np.argsort(-s, kind="stable")
        c, b, s = c[order], b[order], s[order]
        tp = np.zeros(len(s), bool)
        if len(gb) and len(b):
            ious = E.iou(b, gb)
            if class_aware:
                ious = np.where(c[:, None] == gc[None, :], ious, 0)
            taken = np.zeros(len(gb), bool)
            for d in range(len(s)):
                cand = np.where(~taken & (ious[d] >= IOU))[0]
                if len(cand):
                    g = cand[np.argmax(ious[d, cand])]
                    taken[g] = tp[d] = True
        out.append((s, tp, len(gb)))
    return out


def counts(m, thr, idx=None):
    """(caught, n_gt, false alarms, boards) at threshold thr over images idx."""
    idx = range(len(m)) if idx is None else idx
    caught = fa = n_gt = 0
    for i in idx:
        s, tp, g = m[i]
        keep = s >= thr
        caught += int((tp & keep).sum())
        fa += int((~tp & keep).sum())
        n_gt += g
    return caught, n_gt, fa, len(idx)


def thr_for_recall(m, target):
    """Highest threshold whose recall on m is >= target (candidates = every detection score)."""
    for t in np.unique(np.concatenate([x[0] for x in m]))[::-1]:
        c, g, _, _ = counts(m, t)
        if c / g >= target:
            return float(t)
    return 0.0


def thr_for_cost(m, ratio):
    """Threshold minimising ratio * misses + false alarms on m."""
    cands = np.unique(np.concatenate([x[0] for x in m]))
    costs = []
    for t in cands:
        c, g, fa, _ = counts(m, t)
        costs.append(ratio * (g - c) + fa)
    return float(cands[int(np.argmin(costs))])


def flagged(per_clean, thr):
    """Fraction of clean boards with any detection at or above thr."""
    return float(np.mean([(s >= thr).any() for _, _, s in per_clean]))


def run_points(run, class_aware):
    """Everything tests 3 and 4 need for one run."""
    mv = match(*load(run, "val"), class_aware)
    mt = match(*load(run, "test"), class_aware)
    _, clean = load(run, "clean")
    t = thr_for_recall(mv, TARGET_RECALL)
    c, g, fa, n = counts(mt, t)
    r = {"thr99": t, "test_recall": c / g, "fa_per_board": fa / n, "clean_flagged": flagged(clean, t),
         "cost": {}}
    for k in COST_RATIOS:
        tk = thr_for_cost(mv, k)
        c, g, fa, n = counts(mt, tk)
        r["cost"][k] = {"thr": tk, "test_recall": c / g, "fa_per_board": fa / n,
                        "cost_per_board": (k * (g - c) + fa) / n, "clean_flagged": flagged(clean, tk)}
    return r, mt


def fa_boot(mts, thrs, idx):
    """Mean over seeds of false alarms per board, for each bootstrap resample."""
    out = []
    for rs in idx:
        out.append(np.mean([counts(m, t, rs)[2] / len(rs) for m, t in zip(mts, thrs)]))
    return np.array(out)


def main():
    """Judge test 3, report test 4, for both matching variants."""
    global IOU
    ap = argparse.ArgumentParser()
    ap.add_argument("--iou", type=float, default=IOU)
    ap.add_argument("--suffix", default="", help='run-name suffix, e.g. "_v2" (PLAN.md)')
    a = ap.parse_args()
    IOU, suf = a.iou, a.suffix
    if IOU != 0.5:
        print(f"*** POST-HOC SENSITIVITY at IoU {IOU}: not the pre-registered test ***")
    seeds = (0, 1)
    runs = {arm: [f"{arm}900_s{s}{suf}" for s in seeds] for arm in ("A", "T")}
    need = [f"{r}__{sp}.preds.npz" for rs in runs.values() for r in rs for sp in ("val", "test", "clean")]
    if not all((EVAL / f).exists() for f in need):
        print("missing preds for the 900-image runs; nothing to judge")
        return
    report = {}
    for variant, aware in (("class_agnostic", False), ("class_aware", True)):
        res = {arm: [run_points(r, aware) for r in rs] for arm, rs in runs.items()}
        n_test = len(res["A"][0][1])
        idx = np.random.default_rng(4).integers(0, n_test, (N_BOOT, n_test))
        fa = {arm: fa_boot([x[1] for x in res[arm]], [x[0]["thr99"] for x in res[arm]], idx)
              for arm in res}
        d = fa["T"] - fa["A"]
        point = {arm: float(np.mean([x[0]["fa_per_board"] for x in res[arm]])) for arm in res}
        lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
        verdict = "PASS" if hi < 0 else "FAIL"
        rep = {"verdict": verdict, "T_minus_A_fa_per_board": point["T"] - point["A"], "ci95": [lo, hi],
               "per_run": {r: x[0] for arm, rs in runs.items() for r, x in zip(rs, res[arm])}}
        report[variant] = rep
        tag = "TEST 3" if variant == "class_agnostic" else "  (class-aware sensitivity)"
        print(f"{tag} {verdict}: false alarms/board at val-99%-recall threshold, T - A "
              f"{point['T'] - point['A']:+.3f} [{lo:+.3f}, {hi:+.3f}] (A {point['A']:.3f}, T {point['T']:.3f})")
        for r, x in rep["per_run"].items():
            print(f"       {r}: thr {x['thr99']:.3f}  test recall {100 * x['test_recall']:.2f}%  "
                  f"FA/board {x['fa_per_board']:.3f}  clean boards flagged {100 * x['clean_flagged']:.1f}%")
        if variant == "class_agnostic":
            print("TEST 4 (descriptive): threshold minimising ratio*misses + false alarms on val")
            for k in COST_RATIOS:
                for r, x in rep["per_run"].items():
                    y = x["cost"][k]
                    print(f"       {k:>3}:1 {r}: thr {y['thr']:.3f}  recall {100 * y['test_recall']:.2f}%  "
                          f"FA/board {y['fa_per_board']:.3f}  cost/board {y['cost_per_board']:.3f}  "
                          f"clean flagged {100 * y['clean_flagged']:.1f}%")
    name = "operating" + ("" if IOU == 0.5 else f"_iou{str(IOU).replace('.', '')[:3]}") + f"{suf}.json"
    (E.ROOT / "runs" / name).write_text(json.dumps(report, indent=1, default=float))


if __name__ == "__main__":
    main()
