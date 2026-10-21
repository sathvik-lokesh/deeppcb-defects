"""Fine-tune a COCO-pretrained YOLO11s on one arm and training size.

Every run uses the same model, image size, batch and augmentation; only the arm (A or T) and
the training list change (data/arms/<arm><n>.yaml). Smaller training sets are repeated up to
900 entries by prepare.py, so every run takes the same epochs and optimiser steps (see PLAN.md).
Output: runs/<arm><n>_s<seed>/weights/last.pt.

    python train.py --arm A --n 900 --seed 0
    python train.py --arm A --n 900 --seed 0 --resume   # continue an interrupted run
    python train.py --arm A --n 900 --seed 0 --nomosaic # POST-HOC sensitivity, not pre-registered
    python train.py --arm A --n 900 --seed 0 --recipe v2  # v2 deviation (PLAN.md): mosaic=0, runs/<..>_v2
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent

# Fixed for every run (see PLAN.md). Changing any of these means re-running every run.
EPOCHS = 60
HP = dict(model="yolo11s.pt", imgsz=640, batch=16, workers=6, cos_lr=True, amp=True, plots=True,
          val=False,  # no checkpoint selection: val is not needed during training
          hsv_h=0.0, hsv_s=0.0, hsv_v=0.0)  # T's channels carry meaning; images are greyscale anyway


def main():
    """Train one run (or resume it) and print where the weights went."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["A", "T"])
    ap.add_argument("--n", type=int, default=900)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=EPOCHS, help="only for smoke runs")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--nomosaic", action="store_true", help="post-hoc sensitivity: mosaic=0 (PLAN.md)")
    ap.add_argument("--recipe", choices=["v1", "v2"], default="v1", help="v2 = mosaic=0 for every run (PLAN.md)")
    a = ap.parse_args()

    smoke = a.epochs != EPOCHS
    assert not (a.nomosaic and a.recipe == "v2"), "--nomosaic is the v1 post-hoc run; v2 already has mosaic=0"
    suffix = "_nomosaic" if a.nomosaic else "_v2" if a.recipe == "v2" else ""
    name = f"{a.arm}{a.n}_s{a.seed}" + suffix + ("_smoke" if smoke else "")
    run = ROOT / "runs" / name
    if a.resume:
        YOLO(run / "weights" / "last.pt").train(resume=True)
    else:
        hp = {k: v for k, v in HP.items() if k != "model"} | (dict(mosaic=0.0) if suffix else {})
        YOLO(ROOT / "weights" / HP["model"]).train(
            data=str(ROOT / "data" / "arms" / f"{a.arm}{a.n}.yaml"), project=str(ROOT / "runs"),
            name=name, exist_ok=True, seed=a.seed, deterministic=True, epochs=a.epochs,
            close_mosaic=max(1, a.epochs // 12), **hp)
    (run / "DONE").write_text("finished\n")  # queue.sh skips runs that have this
    print(f"done: {run / 'weights' / 'last.pt'}")


if __name__ == "__main__":
    main()
