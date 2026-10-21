"""Check the prepared data before any training: splits, leakage, labels, arm T channels.

    python selftest.py      # exit 1 on any failure
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

import prepare as P

results = []


def check(name, ok, detail=""):
    """Record one named check."""
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))


def lst(arm, split):
    """Image paths of one split list."""
    return (P.SPLITS / arm / f"{split}.txt").read_text().split()


def iid(path):
    """Image id from a prepared image path."""
    return Path(path).stem


def raw_boxes(label_path):
    """Raw DeepPCB boxes as int array [n, 5] (x1 y1 x2 y2 type)."""
    rows = [r.split() for r in label_path.read_text().replace(",", " ").splitlines() if r.strip()]
    return np.array(rows, int).reshape(-1, 5)


def yolo_boxes(img):
    """Prepared YOLO label back to [n, 5] pixels (x1 y1 x2 y2 type 1..6)."""
    lab = Path(img.replace("/images/", "/labels/")).with_suffix(".txt")
    a = np.loadtxt(lab, ndmin=2).reshape(-1, 5) if lab.stat().st_size else np.zeros((0, 5))
    cx, cy, w, h = (a[:, 1:] * P.SIZE).T
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, a[:, 0] + 1], 1)


def main():
    """Run every check; exit 1 if any fails."""
    trainval, test = P.read_list("trainval.txt"), P.read_list("test.txt")
    raw = {s[0]: s for s in trainval + test}

    # 1. sizes and disjointness
    sizes = {s: len(lst("A", s)) for s in ("train", "val", "test", "test_seen", "test_unseen", "clean")}
    check("split sizes", sizes == {"train": 900, "val": 100, "test": 500, "test_seen": 168,
                                   "test_unseen": 332, "clean": 500}, sizes)
    tr, va, te = ({iid(p) for p in lst("A", s)} for s in ("train", "val", "test"))
    check("train/val/test disjoint", not (tr & va or tr & te or va & te))
    check("train + val == official trainval", tr | va == {s[0] for s in trainval})
    check("test == official test", te == {s[0] for s in test})

    # 2. board groups: unseen boards never in train/val, seen + unseen == test
    group = {s[0]: s[1] for s in trainval + test}
    seen_train = {group[i] for i in tr | va}
    unseen = {iid(p) for p in lst("A", "test_unseen")}
    seen = {iid(p) for p in lst("A", "test_seen")}
    check("unseen-board test groups absent from train/val", not ({group[i] for i in unseen} & seen_train))
    check("seen-board test groups all in train", {group[i] for i in seen} <= seen_train)
    check("seen + unseen == test, disjoint", seen | unseen == te and not seen & unseen)
    check("unseen groups == the 4 from the audit", {group[i] for i in unseen} == P.UNSEEN_GROUPS)

    # 3. nested, hash-ordered train subsets
    prev = []
    nested = True
    for n in P.TRAIN_SIZES:
        cur = lst("A", f"train{n}")
        nested &= len(cur) == n and cur[:len(prev)] == prev
        prev = cur
    check("train subsets nested and sized", nested)
    g25 = {group[iid(p)] for p in lst("A", "train25")}
    check("train25 spans several boards (not first-N lines)", len(g25) >= 4, f"{len(g25)} groups")
    rep_ok = True
    for arm in P.ARMS:
        for n in P.TRAIN_SIZES:
            r = lst(arm, f"train{n}_rep")
            base = lst(arm, f"train{n}")
            stems = [iid(x).split("__r")[0] for x in r]
            rep_ok &= (len(r) == 900 and len(set(r)) == 900 and sorted(set(stems)) == sorted(iid(x) for x in base)
                       and all(stems.count(iid(x)) == 900 // n for x in base[:5])
                       and all(os.path.samefile(x, y) for x, y in zip(r[:n], base)))
    check("repeated subsets: 900 distinct names, each image 900/n times, same files", rep_ok)

    # 4. A and T lists aligned image by image (paired comparisons need this)
    aligned = all([iid(a) for a in lst("A", s)] == [iid(t) for t in lst("T", s)]
                  for s in ("train", "val", "test", "test_seen", "test_unseen", "clean"))
    check("A and T split lists aligned", aligned)

    # 5. labels round-trip and box totals
    worst, n_tv, n_te = 0.0, 0, 0
    for arm in P.ARMS:
        for s in ("train", "val", "test"):
            for p in lst(arm, s):
                rb, yb = raw_boxes(raw[iid(p)][4]), yolo_boxes(p)
                ok = rb.shape == yb.shape
                worst = max(worst, np.abs(rb - yb).max() if ok and len(rb) else (0 if ok else 1e9))
                if arm == "A":
                    n_te += len(rb) if s == "test" else 0
                    n_tv += len(rb) if s != "test" else 0
    check("labels round-trip to raw pixels (< 0.01 px)", worst < 0.01, f"max err {worst:.4f}")
    check("box totals match the audit (6873 / 3140)", (n_tv, n_te) == (6873, 3140), (n_tv, n_te))
    clean_empty = all(Path(p.replace("/images/", "/labels/")).with_suffix(".txt").stat().st_size == 0
                      for arm in P.ARMS for p in lst(arm, "clean"))
    check("clean boards have no labels", clean_empty)

    # 6. arm A images are the raw tested files (hard links); clean A = raw template
    a_ok = all(os.path.samefile(p, raw[iid(p)][2]) for s in ("train", "val", "test") for p in lst("A", s))
    c_ok = all(os.path.samefile(p, raw[iid(p)[:-len("_clean")]][3]) for p in lst("A", "clean"))
    check("A images == raw tested files", a_ok)
    check("A clean images == raw templates", c_ok)

    # 7. arm T channels, on every image
    t_bad = c_bad = 0
    for s in ("train", "val", "test"):
        for p in lst("T", s):
            img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
            a, b = P.gray(raw[iid(p)][2]), P.gray(raw[iid(p)][3])
            t_bad += not (img.shape == (640, 640, 3) and (img[..., 0] == a).all() and (img[..., 1] == b).all()
                          and (img[..., 2] == cv2.absdiff(a, b)).all())
    for p in lst("T", "clean"):
        img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        b = P.gray(raw[iid(p)[:-len("_clean")]][3])
        c_bad += not ((img[..., 0] == b).all() and (img[..., 1] == b).all() and (img[..., 2] == 0).all())
    check("T channels == (tested, template, |diff|) on all 1500", t_bad == 0, f"{t_bad} bad")
    check("T clean == (template, template, 0) on all 500", c_bad == 0, f"{c_bad} bad")
    # Ultralytics reads with cv2 (BGR) and flips to RGB: the model sees the same 3 planes reversed,
    # identically for every image, so nothing to fix. Confirm cv2's default read keeps the planes.
    p = lst("T", "test")[0]
    check("cv2 default read keeps T planes", (cv2.imread(p) == cv2.imread(p, cv2.IMREAD_UNCHANGED)).all())
    check("A jpg reads as 3 equal planes", (lambda x: (x[..., 0] == x[..., 1]).all()
                                             and (x[..., 1] == x[..., 2]).all())(cv2.imread(lst("A", "test")[0])))

    # 8. manifest hashes match the lists on disk
    m = json.loads((P.ROOT / "data" / "manifest.json").read_text())
    bad = [k for k, v in m["splits"].items()
           if hashlib.sha256((P.SPLITS / f"{k}.txt").read_bytes()).hexdigest() != v["sha256"]]
    check("manifest sha256 == split lists", not bad, bad)

    print(f"\n{sum(results)}/{len(results)} checks passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
