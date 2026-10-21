"""Python 3 port of DeepPCB's evaluation/script.py (ICDAR 2015 code), for numbers comparable to the paper.

    python official_eval.py --run A900_s0          # needs runs/eval/<run>__{val,test}.preds.npz

The shipped script cannot run as is: it is Python 2.7, needs the `Polygon` module, and its parser
expects comma-separated 8-point lines with class-name strings while the shipped gt.zip holds
space-separated 4-number lines with class ids. So this ports its logic instead, read line by line:
- detections of an image sorted by confidence, descending (sort_by_confidences=True);
- matching loops ground truth first, then detections in that order; a pair matches when both
  are unused and IoU > 0.33 (strict), class ignored; the match is "correct" only if the class
  agrees, but a wrong-class match still uses up both boxes;
- AP = one number over all classes: sum over correct detections (global confidence order) of
  precision at that rank, divided by the number of gt boxes (no interpolation);
- precision / recall / F (hmean) use every detection submitted, so they depend on the score
  threshold. The README of DeepPCB says to tune it; we choose it on val (max F) and apply it to test,
  and also print the test-oracle max F, which is what a tuned-on-test number would be.
Ground truth is read from the shipped evaluation/gt.zip and checked equal to our labels.
"""
import argparse
import json
import zipfile

import numpy as np

import evaluate as E
import prepare as P

GT_ZIP = P.RAW.parent / "evaluation" / "gt.zip"
IOU = 0.33


def gt_from_zip():
    """{image id: int array [n, 5] (x1 y1 x2 y2 type 1..6)} from the shipped gt.zip."""
    z = zipfile.ZipFile(GT_ZIP)
    out = {}
    for name in z.namelist():
        rows = [r.split() for r in z.read(name).decode().replace(",", " ").splitlines() if r.strip()]
        out[name[:-len(".txt")]] = np.array(rows, int).reshape(-1, 5)
    return out


def per_image(gt, det_cls, det_box, det_score):
    """Port of the per-sample loop: (confidences, correct flags, n_gt, n_correct) for one image."""
    order = np.argsort(-det_score, kind="stable")
    det_cls, det_box, det_score = det_cls[order], det_box[order], det_score[order]
    gt_used = np.zeros(len(gt), bool)
    det_used = np.zeros(len(det_box), bool)
    correct = np.zeros(len(det_box), bool)
    if len(gt) and len(det_box):
        ious = E.iou(gt[:, :4].astype(float), det_box)
        for g in range(len(gt)):
            for d in range(len(det_box)):
                if not gt_used[g] and not det_used[d] and ious[g, d] > IOU:
                    gt_used[g] = det_used[d] = True
                    correct[d] = det_cls[d] + 1 == gt[g, 4]
    return det_score, correct, len(gt), int(correct.sum())


def compute_ap(conf, match, n_gt):
    """Port of compute_ap: uninterpolated, pooled over classes."""
    order = np.argsort(-conf, kind="stable")
    m = match[order]
    ranks = np.arange(1, len(m) + 1)
    return float((np.cumsum(m)[m] / ranks[m]).sum() / n_gt) if n_gt else 0.0


def prf(samples, thr):
    """Method precision, recall, hmean over detections with score >= thr."""
    tp = n_det = n_gt = 0
    for s, c, g, _ in samples:
        keep = s >= thr
        tp += int((c & keep).sum())
        n_det += int(keep.sum())
        n_gt += g
    p = tp / n_det if n_det else 0.0
    r = tp / n_gt if n_gt else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def samples_for(run, split, gt):
    """Port's per-image results for one run on one split."""
    z = np.load(E.ROOT / "runs" / "eval" / f"{run}__{split}.preds.npz")
    out = []
    for i, img in enumerate(z["images"]):
        k = z["img"] == i
        out.append(per_image(gt[img], z["cls"][k], z["xyxy"][k], z["score"][k]))
    return out


def best_thr(samples):
    """Score threshold with the highest hmean (candidates on a 0.001 grid)."""
    grid = np.round(np.arange(0.001, 1.0, 0.001), 3)
    f = [prf(samples, t)[2] for t in grid]
    return float(grid[int(np.argmax(f))])


def main():
    """Official-protocol AP and F for one run; checks the shipped gt equals our labels first."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    zgt = gt_from_zip()
    raw = {s[0]: s for s in P.read_list("trainval.txt") + P.read_list("test.txt")}
    test_ids = [s[0] for s in P.read_list("test.txt")]
    assert sorted(zgt) == sorted(test_ids), "gt.zip images != official test list"
    for iid in test_ids:
        ours = np.array([r.split() for r in raw[iid][4].read_text().replace(",", " ").splitlines() if r.strip()],
                        int).reshape(-1, 5)
        assert (ours == zgt[iid]).all(), f"gt.zip differs from our labels on {iid}"
    # val has no gt.zip: use our raw labels (same files, same format)
    gt = dict(zgt)
    for s in P.read_list("trainval.txt"):
        gt[s[0]] = np.array([r.split() for r in s[4].read_text().replace(",", " ").splitlines() if r.strip()],
                            int).reshape(-1, 5)
    gt_by_path = {}
    for split in ("val", "test"):
        for p in (P.SPLITS / a.run[0] / f"{split}.txt").read_text().split():
            gt_by_path[p] = gt[Path_stem(p)]
    sv, st = samples_for(a.run, "val", gt_by_path), samples_for(a.run, "test", gt_by_path)
    conf = np.concatenate([s[0] for s in st])
    match = np.concatenate([s[1] for s in st])
    n_gt = sum(s[2] for s in st)
    res = {"run": a.run, "AP": compute_ap(conf, match, n_gt)}
    t = best_thr(sv)
    res["val_thr"], (res["P"], res["R"], res["F"]) = t, prf(st, t)
    to = best_thr(st)
    res["oracle_thr"], res["oracle_F"] = to, prf(st, to)[2]
    out = E.ROOT / "runs" / "eval" / f"{a.run}__official.json"
    out.write_text(json.dumps(res, indent=1))
    print(f"{a.run} official protocol (port): AP {100 * res['AP']:.2f}  F {100 * res['F']:.2f} "
          f"(P {100 * res['P']:.2f} R {100 * res['R']:.2f}, val thr {t:.3f})  "
          f"test-oracle F {100 * res['oracle_F']:.2f}  | paper: 98.6 mAP, 98.2 F")


def Path_stem(p):
    """Image id from a prepared image path."""
    return p.rsplit("/", 1)[1].rsplit(".", 1)[0]


if __name__ == "__main__":
    main()
