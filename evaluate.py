"""Score a detector on one split with COCO-style mAP, keeping per-image match data for bootstrap CIs.

    python evaluate.py --weights runs/A900_s0/weights/last.pt --split test

The arm (A or T) is read from the run name and picks the matching image lists. Writes
runs/eval/<tag>__<split>.npz (per-image matches), <tag>__<split>.preds.npz (raw detections,
for operating points and the clean-board set) and a json summary. On `test` the summary also
covers the seen-board and unseen-board subsets. `clean` has no ground truth: preds only.

Copied from bdd-yolo-robustness (verified == pycocotools there, re-checked by selftest_eval.py).
Changes: 640x640 images, 6 classes, an extra IoU 0.33 threshold (the official DeepPCB criterion)
stored in front of the COCO ones, and only the "all" area range.

The matching follows pycocotools (greedy by score, a det takes the best-IoU unmatched gt,
101-point interpolated precision, maxDets 100) and is checked against it in selftest_eval.py.
"""
import argparse
import json
from pathlib import Path

import numpy as np

import prepare as P

ROOT = P.ROOT
IOUS = np.r_[0.33, np.linspace(0.5, 0.95, 10)]  # index 0 = DeepPCB criterion, 1: = COCO 50-95
COCO = slice(1, None)
REC = np.linspace(0, 1, 101)
# COCO size buckets; DeepPCB boxes are mostly small/medium, so only "all" is scored by default
AREAS = {"all": (0, 1e10), "small": (0, 32 ** 2), "medium": (32 ** 2, 96 ** 2), "large": (96 ** 2, 1e10)}
SCORED_AREAS = ("all",)
W = H = P.SIZE
MAX_DETS = 100


def load_gt(img):
    """Ground-truth boxes of one image as (cls[n], xyxy[n,4]) in pixels."""
    lab = Path(img.replace("/images/", "/labels/")).with_suffix(".txt")
    a = np.loadtxt(lab, ndmin=2) if lab.stat().st_size else np.zeros((0, 5))
    cx, cy, w, h = a[:, 1] * W, a[:, 2] * H, a[:, 3] * W, a[:, 4] * H
    return a[:, 0].astype(int), np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)


def iou(a, b):
    """Pairwise IoU between boxes a[n,4] and b[m,4] (xyxy)."""
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(rb - lt, 0, None), 2)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-12)


def match_image(gt_box, dt_box, dt_score, rng):
    """COCO matching for one image and one class within one area range.

    Returns (scores[d], matched[T,d] bool, ignored[T,d] bool, n_gt_counted).
    """
    lo, hi = rng
    garea = (gt_box[:, 2] - gt_box[:, 0]) * (gt_box[:, 3] - gt_box[:, 1])
    gign = (garea < lo) | (garea > hi)
    gorder = np.argsort(gign, kind="stable")  # counted gts first, as pycocotools does
    gt_box, gign = gt_box[gorder], gign[gorder]
    dorder = np.argsort(-dt_score, kind="stable")[:MAX_DETS]
    dt_box, dt_score = dt_box[dorder], dt_score[dorder]
    T, D, G = len(IOUS), len(dt_box), len(gt_box)
    matched = np.zeros((T, D), bool)
    ignored = np.zeros((T, D), bool)
    if G and D:
        ious = iou(dt_box, gt_box)
        for t, thr in enumerate(IOUS):
            taken = np.zeros(G, bool)
            for d in range(D):
                best, m = min(thr, 1 - 1e-10), -1
                for g in range(G):
                    if taken[g]:
                        continue
                    if m > -1 and not gign[m] and gign[g]:
                        break  # already matched a counted gt; never switch to an ignored one
                    if ious[d, g] < best:
                        continue
                    best, m = ious[d, g], g
                if m > -1:
                    taken[m] = True
                    matched[t, d] = True
                    ignored[t, d] = gign[m]
    darea = (dt_box[:, 2] - dt_box[:, 0]) * (dt_box[:, 3] - dt_box[:, 1])
    out = (darea < lo) | (darea > hi)
    ignored |= ~matched & out[None, :]  # unmatched dets outside the range do not count
    return dt_score, matched, ignored, int((~gign).sum())


def ap_from(scores, matched, ignored, n_gt):
    """101-point interpolated AP per IoU threshold from concatenated match data; nan if no gt."""
    if n_gt == 0:
        return np.full(len(IOUS), np.nan)
    order = np.argsort(-scores, kind="mergesort")
    m, ig = matched[:, order], ignored[:, order]
    tp = np.cumsum(m & ~ig, 1)
    fp = np.cumsum(~m & ~ig, 1)
    out = np.zeros(len(IOUS))
    for t in range(len(IOUS)):
        if tp.shape[1] == 0:
            continue
        rc = tp[t] / n_gt
        pr = tp[t] / np.maximum(tp[t] + fp[t], np.spacing(1))
        pr = np.maximum.accumulate(pr[::-1])[::-1]
        idx = np.searchsorted(rc, REC, side="left")
        q = np.zeros(len(REC))
        ok = idx < len(pr)
        q[ok] = pr[idx[ok]]
        out[t] = q.mean()
    return out


class Matches:
    """Per-image, per-class match data for one (model, split), with mAP and bootstrap helpers."""

    def __init__(self, per_image, classes):
        self.per_image, self.classes = per_image, classes  # per_image[i][area][cls] = tuple

    def map(self, idx=None, area="all", classes=None):
        """(mAP50-95, mAP50, per-class AP50-95) over images idx (with repeats allowed)."""
        idx = range(len(self.per_image)) if idx is None else idx
        classes = classes or self.classes
        aps = {}
        for c in classes:
            parts = [self.per_image[i][area][c] for i in idx]
            s = np.concatenate([p[0] for p in parts])
            m = np.concatenate([p[1] for p in parts], 1)
            ig = np.concatenate([p[2] for p in parts], 1)
            aps[c] = ap_from(s, m, ig, sum(p[3] for p in parts))
        arr = np.array([aps[c] for c in classes])
        return (np.nanmean(arr[:, COCO]), np.nanmean(arr[:, 1]),
                {c: np.nanmean(aps[c][COCO]) for c in classes}, np.nanmean(arr[:, 0]))

    def bootstrap(self, n=1000, seed=0, area="all", classes=None):
        """mAP50-95 for n image resamples (fixed seed -> reproducible CIs)."""
        r = np.random.default_rng(seed)
        N = len(self.per_image)
        return np.array([self.map(r.integers(0, N, N), area, classes)[0] for _ in range(n)])

    def save(self, path):
        """Store as an object array in a .npz."""
        np.savez_compressed(path, per_image=np.array(self.per_image, dtype=object),
                            classes=np.array(self.classes))

    @classmethod
    def load(cls, path):
        """Inverse of save."""
        z = np.load(path, allow_pickle=True)
        return cls(list(z["per_image"]), [int(c) for c in z["classes"]])


def build(images, preds, classes, areas=SCORED_AREAS):
    """Match predictions to ground truth for every image, class and area range."""
    per_image = []
    for img in images:
        gc, gb = load_gt(img)
        pc, pb, ps = preds[img]
        per_image.append({a: {c: match_image(gb[gc == c], pb[pc == c], ps[pc == c], AREAS[a])
                              for c in classes} for a in areas})
    return Matches(per_image, classes)


def predict(weights, images, imgsz=640, batch=32):
    """Run the detector; returns {img: (cls, xyxy, score)} in our class ids, pixels."""
    from ultralytics import YOLO
    model = YOLO(weights)
    out = {}
    for i in range(0, len(images), batch):
        chunk = images[i:i + batch]
        for img, r in zip(chunk, model.predict(chunk, imgsz=imgsz, conf=0.001, iou=0.7, max_det=300,
                                               verbose=False)):
            c = r.boxes.cls.cpu().numpy().astype(int)
            b = r.boxes.xyxy.cpu().numpy()
            s = r.boxes.conf.cpu().numpy()
            out[img] = (c, b, s)
    return out


def summary(m, idx=None, n_boot=1000):
    """Dict with mAP50-95, its 95% CI, mAP50, AP@0.33 and per-class AP50-95 over images idx."""
    mp, m50, per, m33 = m.map(idx)
    r = np.random.default_rng(0)
    idx = np.arange(len(m.per_image)) if idx is None else np.asarray(idx)
    boot = np.array([m.map(idx[r.integers(0, len(idx), len(idx))])[0] for _ in range(n_boot)])
    return {"mAP50_95": mp, "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "mAP50": m50, "AP33": m33, "per_class": {P.CLASSES[c]: v for c, v in per.items()},
            "images": int(len(idx))}


def subset_index(arm, split, sub):
    """Positions of the images of list `sub` inside list `split` (both of one arm)."""
    pos = {p: i for i, p in enumerate((P.SPLITS / arm / f"{split}.txt").read_text().split())}
    return [pos[p] for p in (P.SPLITS / arm / f"{sub}.txt").read_text().split()]


def save_preds(path, images, preds):
    """Raw detections as flat arrays: image index, class, xyxy, score."""
    ii = np.concatenate([np.full(len(preds[im][0]), i) for i, im in enumerate(images)])
    np.savez_compressed(path, images=np.array(images), img=ii,
                        cls=np.concatenate([preds[im][0] for im in images]),
                        xyxy=np.concatenate([preds[im][1] for im in images]).reshape(-1, 4),
                        score=np.concatenate([preds[im][2] for im in images]))


def main():
    """Predict, save raw detections, then match + summarise (except on `clean`)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--split", required=True, choices=["val", "test", "clean"])
    ap.add_argument("--tag", help="name for outputs (default: run folder name)")
    ap.add_argument("--imgsz", type=int, default=640)
    a = ap.parse_args()
    w = Path(a.weights)
    tag = a.tag or w.parent.parent.name
    arm = tag[0]
    assert arm in P.ARMS, f"run name must start with the arm letter: {tag}"
    images = (P.SPLITS / arm / f"{a.split}.txt").read_text().split()
    preds = predict(str(w), images, a.imgsz)
    out = ROOT / "runs" / "eval"
    out.mkdir(parents=True, exist_ok=True)
    save_preds(out / f"{tag}__{a.split}.preds.npz", images, preds)
    if a.split == "clean":
        print(f"{tag} clean: {sum(len(v[0]) for v in preds.values())} detections saved")
        return
    m = build(images, preds, list(range(len(P.CLASSES))))
    m.save(out / f"{tag}__{a.split}.npz")
    s = {"tag": tag, "split": a.split, "weights": str(w), "main": summary(m)}
    if a.split == "test":
        for sub in ("test_seen", "test_unseen"):
            s[sub] = summary(m, subset_index(arm, "test", sub))
    (out / f"{tag}__{a.split}.json").write_text(json.dumps(s, indent=1, default=float))
    for k in ("main", "test_seen", "test_unseen"):
        if k in s:
            r = s[k]
            print(f"{tag} {a.split} {k}: mAP50-95 {100 * r['mAP50_95']:.2f} "
                  f"[{100 * r['ci95'][0]:.2f}, {100 * r['ci95'][1]:.2f}]  mAP50 {100 * r['mAP50']:.2f}"
                  f"  AP@.33 {100 * r['AP33']:.2f}")


if __name__ == "__main__":
    main()
