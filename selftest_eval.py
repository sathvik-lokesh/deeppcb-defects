"""Check evaluate.py's AP against pycocotools on DeepPCB ground truth.

Detections are synthetic (jittered gt boxes, some with the wrong class, plus random false
positives, random scores), so no trained model is needed and every matching path is exercised.
Requires agreement to 1e-6 for COCO mAP50-95, mAP50 and AP at IoU 0.33, on all 500 test images
and on a subset index (the unseen boards).
"""
import contextlib
import io
import sys

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import evaluate as E
import prepare as P


def synthetic_preds(images, seed=0):
    """Plausible detections: most gts found with jitter, some misclassified, some missed, plus noise."""
    r = np.random.default_rng(seed)
    out = {}
    for img in images:
        gc, gb = E.load_gt(img)
        keep = r.random(len(gc)) < 0.85
        c = np.where(r.random(len(gc)) < 0.1, r.integers(0, 6, len(gc)), gc)[keep]
        b = gb[keep] + r.normal(0, 4, (keep.sum(), 4))
        nf = r.integers(0, 6)
        xy = r.uniform(0, 600, (nf, 2))
        fb = np.c_[xy, xy + r.uniform(15, 60, (nf, 2))]
        c = np.r_[c, r.integers(0, 6, nf)].astype(int)
        b = np.clip(np.r_[b, fb], 0, 640)
        out[img] = (c, b, r.random(len(c)))
    return out


def coco_reference(images, preds, iou_thrs):
    """Mean over classes of AP averaged over iou_thrs, from pycocotools."""
    classes = list(range(len(P.CLASSES)))
    gt = {"images": [], "annotations": [], "categories": [{"id": c} for c in classes]}
    dt = []
    for i, img in enumerate(images):
        gt["images"].append({"id": i, "width": E.W, "height": E.H})
        gc, gb = E.load_gt(img)
        for c, b in zip(gc, gb):
            w, h = b[2] - b[0], b[3] - b[1]
            gt["annotations"].append({"id": len(gt["annotations"]) + 1, "image_id": i, "category_id": int(c),
                                      "bbox": [b[0], b[1], w, h], "area": w * h, "iscrowd": 0})
        pc, pb, ps = preds[img]
        for c, b, s in zip(pc, pb, ps):
            dt.append({"image_id": i, "category_id": int(c), "score": float(s),
                       "bbox": [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])]})
    with contextlib.redirect_stdout(io.StringIO()):
        g = COCO()
        g.dataset = gt
        g.createIndex()
        ev = COCOeval(g, g.loadRes(dt), "bbox")
        ev.params.iouThrs = np.array(iou_thrs)
        ev.params.areaRng = [list(E.AREAS["all"])]
        ev.params.areaRngLbl = ["all"]
        ev.evaluate()
        ev.accumulate()
    p = ev.eval["precision"][:, :, :, 0, -1]  # T, R, K at maxDets=100
    return float(np.mean([p[:, :, k].mean() for k in range(p.shape[2]) if (p[:, :, k] > -1).all()]))


def main():
    """Compare both implementations on the full test set and on the unseen-board subset."""
    ok = True
    test = (P.SPLITS / "A" / "test.txt").read_text().split()
    preds = synthetic_preds(test)
    m = E.build(test, preds, list(range(len(P.CLASSES))))
    sub = E.subset_index("A", "test", "test_unseen")
    for name, idx, imgs in (("test", None, test), ("test_unseen", sub, [test[i] for i in sub])):
        mine = m.map(idx)
        refs = {"mAP50-95": (mine[0], coco_reference(imgs, preds, list(E.IOUS[E.COCO]))),
                "mAP50": (mine[1], coco_reference(imgs, preds, [0.5])),
                "AP@.33": (mine[3], coco_reference(imgs, preds, [0.33]))}
        for k, (a, b) in refs.items():
            good = abs(a - b) < 1e-6
            ok &= good
            print(f"{'PASS' if good else 'FAIL'}  {name:11s} {k:8s} ours {a:.6f}  pycocotools {b:.6f}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
