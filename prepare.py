"""Convert DeepPCB into YOLO format for both arms, plus fixed split lists.

Reads  data/raw/DeepPCB/PCBData/{trainval,test}.txt and the images/labels they name
Writes data/yolo/A/images/<id>.jpg      tested image (hard link, YOLO reads it as 3 equal channels)
       data/yolo/T/images/<id>.png      3 channels = (tested, template, |tested - template|)
       data/yolo/{A,T}/images/<id>_clean.*   defect-free template as a clean board (test split only)
       data/yolo/{A,T}/labels/<id>[_clean].txt   (class cx cy w h, normalised; empty for clean)
       data/splits/<arm>/<split>.txt    absolute image paths, one per line
       data/arms/<arm><n>.yaml          Ultralytics dataset files (train on n images, val = val)
       data/manifest.json               counts, groups, sha256 of every split list

Splits are decided by sha1(image id), never by RNG state, so they never move.
"""
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw" / "DeepPCB" / "PCBData"
YOLO = ROOT / "data" / "yolo"
SPLITS = ROOT / "data" / "splits"
ARMS_DIR = ROOT / "data" / "arms"

# DeepPCB type ids 1..6 -> our class ids 0..5 (0 = background is unused in the labels).
CLASSES = ["open", "short", "mousebite", "spur", "copper", "pin-hole"]
ARMS = ("A", "T")
EXT = {"A": "jpg", "T": "png"}
SIZE = 640
N_VAL = 100
TRAIN_SIZES = [25, 50, 100, 225, 450, 900]  # nested, hash order; each divides 900 (see repeat())
UNSEEN_GROUPS = {"12000", "12100", "12300", "90100"}  # in test only (audit 2026-10-20)


def name_hash(name):
    """Stable sort key for an image id, independent of Python's hash seed."""
    return hashlib.sha1(name.encode()).hexdigest()


def read_list(name):
    """Official split list -> [(id, group, tested_path, template_path, label_path)]."""
    out = []
    for line in (RAW / name).read_text().splitlines():
        if not line.strip():
            continue
        img, lab = line.split()
        base = RAW / img[:-len(".jpg")]
        iid = base.name
        out.append((iid, img.split("/")[0][len("group"):], Path(f"{base}_test.jpg"),
                    Path(f"{base}_temp.jpg"), RAW / lab))
    return out


def to_yolo_lines(label_path):
    """DeepPCB 'x1 y1 x2 y2 type' rows (pixels) -> YOLO label lines."""
    lines = []
    for row in label_path.read_text().replace(",", " ").splitlines():
        if not row.strip():
            continue
        x1, y1, x2, y2, t = map(int, row.split())
        assert 1 <= t <= 6 and 0 <= x1 < x2 <= SIZE and 0 <= y1 < y2 <= SIZE, (label_path, row)
        lines.append(f"{t - 1} {(x1 + x2) / 2 / SIZE:.6f} {(y1 + y2) / 2 / SIZE:.6f} "
                     f"{(x2 - x1) / SIZE:.6f} {(y2 - y1) / SIZE:.6f}")
    return lines


def gray(path):
    """Read one greyscale image as uint8 [640, 640]."""
    a = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    assert a is not None and a.shape == (SIZE, SIZE), path
    return a


def t_image(tested, template):
    """Arm T input: (tested, template, |tested - template|) stacked as 3 channels."""
    a, b = gray(tested), gray(template)
    return np.dstack([a, b, cv2.absdiff(a, b)])


def write_sample(arm, name, tested, template, lines):
    """Write one image + label for an arm; returns the image path. Skips images already written."""
    img = YOLO / arm / "images" / f"{name}.{EXT[arm]}"
    if not img.exists():
        if arm == "A":
            os.link(tested, img)
        else:
            assert cv2.imwrite(str(img), t_image(tested, template))
    (YOLO / arm / "labels" / f"{name}.txt").write_text("".join(l + "\n" for l in lines))
    return str(img)


def repeat(arm, paths):
    """Repeat a training subset to 900 entries via hard links with distinct names.

    Ultralytics keys its label cache by file name, so listing a path twice would be collapsed;
    distinct names keep every run at 900 images per epoch, i.e. the same steps and epochs.
    """
    k = 900 // len(paths)
    assert k * len(paths) == 900
    if k == 1:
        return list(paths)
    out = []
    for r in range(k):
        for p in paths:
            src = Path(p)
            for d, ext in (("images", src.suffix), ("labels", ".txt")):
                s = src if d == "images" else Path(p.replace("/images/", "/labels/")).with_suffix(".txt")
                dst = YOLO / arm / "rep" / d / f"{src.stem}__r{r}{ext}"
                if not dst.exists():
                    os.link(s, dst)
            out.append(str(YOLO / arm / "rep" / "images" / f"{src.stem}__r{r}{src.suffix}"))
    return out


def write_list(path, paths):
    """Write one absolute path per line and return the sha256 of the file."""
    text = "".join(f"{p}\n" for p in paths)
    path.write_text(text)
    return hashlib.sha256(text.encode()).hexdigest()


def write_yaml(path, train, val):
    """Ultralytics dataset yaml (Ultralytics scores the 'val' entry when evaluating)."""
    names = "\n".join(f"  {i}: {c}" for i, c in enumerate(CLASSES))
    path.write_text(f"path: {ROOT / 'data'}\ntrain: {train}\nval: {val}\nnames:\n{names}\n")


def main():
    """Build both arms' YOLO layouts, split lists, yamls and the manifest."""
    trainval, test = read_list("trainval.txt"), read_list("test.txt")
    assert (len(trainval), len(test)) == (1000, 500)
    trainval.sort(key=lambda s: name_hash(s[0]))
    splits = {"val": trainval[:N_VAL], "train": trainval[N_VAL:], "test": test}
    for arm in ARMS:
        for d in ("images", "labels", "rep/images", "rep/labels"):
            (YOLO / arm / d).mkdir(parents=True, exist_ok=True)
        (SPLITS / arm).mkdir(parents=True, exist_ok=True)
    ARMS_DIR.mkdir(parents=True, exist_ok=True)

    manifest = {"source": "github.com/tangsanli5201/DeepPCB@08e98c4db5922613fb97176eb3d6497d48260cb1",
                "classes": CLASSES, "splits": {}, "groups": {}}
    for arm in ARMS:
        lists = {}
        for split, items in splits.items():
            lists[split] = [write_sample(arm, iid, te, tp, to_yolo_lines(lab)) for iid, _, te, tp, lab in items]
        lists["test_seen"] = [p for p, s in zip(lists["test"], test) if s[1] not in UNSEEN_GROUPS]
        lists["test_unseen"] = [p for p, s in zip(lists["test"], test) if s[1] in UNSEEN_GROUPS]
        # Clean boards: the template alone, no defects. For T that is (template, template, 0).
        lists["clean"] = [write_sample(arm, f"{iid}_clean", tp, tp, []) for iid, _, _, tp, _ in test]
        for n in TRAIN_SIZES:
            lists[f"train{n}"] = lists["train"][:n]  # nested: train25 is inside train50
            lists[f"train{n}_rep"] = repeat(arm, lists[f"train{n}"])
        for split, paths in lists.items():
            sha = write_list(SPLITS / arm / f"{split}.txt", paths)
            manifest["splits"][f"{arm}/{split}"] = {"images": len(paths), "sha256": sha}
        for n in TRAIN_SIZES:
            write_yaml(ARMS_DIR / f"{arm}{n}.yaml", SPLITS / arm / f"train{n}_rep.txt", SPLITS / arm / "val.txt")
        for split in ("val", "test", "test_seen", "test_unseen", "clean"):
            write_yaml(ARMS_DIR / f"eval_{arm}_{split}.yaml", SPLITS / arm / "train.txt", SPLITS / arm / f"{split}.txt")
    for split, items in splits.items():
        groups = {}
        for s in items:
            groups[s[1]] = groups.get(s[1], 0) + 1
        manifest["groups"][split] = dict(sorted(groups.items()))
    (ROOT / "data" / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: v["images"] for k, v in manifest["splits"].items() if k.startswith("A/")}))
    print(json.dumps(manifest["groups"]))


if __name__ == "__main__":
    main()
