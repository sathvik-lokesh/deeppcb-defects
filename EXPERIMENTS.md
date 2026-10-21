# Experiment design

The tests, splits and decisions for this project, written before the results they apply to. Results are in [README.md](README.md).

## Question
A factory wants to catch PCB defects. YOLO on this dataset is already near the published 98.6%
mAP, so the score alone says nothing new. The questions are:
1. Does it hold up on **boards it never saw** in training?
2. Does giving the model the **defect-free template** of the same board (what real inspection
   lines compare against) help, and does it cut false alarms?
3. Where should the confidence threshold sit when **a missed defect costs far more than a false
   alarm**?
4. How many labelled boards do you actually need?

## Data (fixed before any result)
DeepPCB, github.com/tangsanli5201/DeepPCB at commit `08e98c4d` (2018-12-19), shallow clone into
`data/raw/DeepPCB` (132 MB data + 87 MB .git). LICENSE file = MIT, but the README also says
"You can only use this dataset for research purpose": treat it as research use, say so in the README.

- 1,500 pairs of 640x640 binarised greyscale JPEGs: `<id>_test.jpg` (with defects) and
  `<id>_temp.jpg` (aligned defect-free template). Labels `<id>.txt`, one `x1,y1,x2,y2,type`
  per line (pixel corners), types 1 open, 2 short, 3 mousebite, 4 spur, 5 copper, 6 pin-hole.
- Official split: trainval 1,000, test 500 (lists in `PCBData/*.txt`, no trailing newline).
  Audit 2026-10-20: 0 missing files, 0 out-of-range boxes, 0 duplicate tested images across
  splits, all images 640x640.
- Boxes: trainval 6,873 (open 1283, short 1028, mousebite 1379, spur 1142, copper 1010,
  pin-hole 1031); test 3,140 (659/478/586/483/464/470). 1-15 boxes per image (mean 6.7).
  Box long side p5/p50/p95 = 29/38/70 px: small objects, but not tiny at 640.
- **Board groups** (11 groups = 11 source boards, cropped from ~16k x 16k scans):

| group | trainval | test | test subset |
|---|---|---|---|
| 00041, 13000, 20085, 44000, 50600, 77000, 92000 | 1000 | 168 | **seen boards** |
| 12000, 12100, 12300, 90100 | 0 | 332 | **unseen boards** |

### Splits we make
- `val`: 100 of trainval, chosen by hash of the image id (stable, no RNG state). Used for
  monitoring and for **choosing thresholds** (never the test set).
- `train`: the other 900. Data-curve subsets are nested, by the same hash order:
  25, 50, 100, 225, 450, 900 (each repeated to 900 entries per epoch, see below).
- `test`: official 500, reported as full / seen boards / unseen boards.
- `clean`: the 500 test-split **template** images, with no defects. A board counts as flagged if
  any detection is at or above the threshold. This measures board-level false alarms, which the
  tested images cannot (every one has a defect).

### Arms
- **A (image only)**: tested image copied into all 3 channels.
- **T (template)**: 3 channels = (tested, template, |tested - template|). Same YOLO, no
  architecture change. Clean boards for T = (template, template, 0).
- Both arms: YOLO11s from COCO weights, imgsz 640, same epochs and seeds, colour augmentation off
  (`hsv_h=hsv_s=hsv_v=0`: the channels mean something in T and the images are greyscale anyway).
  Geometric augmentation (flip, mosaic, scale) is applied to all channels together, so it is kept.
- **No checkpoint selection**: every run is scored at its last epoch.
- Seeds: 2 per arm (s0, s1) for the full-data runs and the data-curve points at 25/50/100;
  1 seed (s0) at 225 and 450.

## Metrics
- Primary: COCO mAP50-95 (BDD `evaluate.py`), image-bootstrap 95% CI, paired across arms.
- Also: the **official DeepPCB protocol** (IoU > 0.33 per class, mAP + F-score), so our numbers
  compare with the paper's 98.6% mAP / 98.2% F. Check that our IoU-0.33 evaluator matches the
  official `evaluation/script.py` on one run before quoting it. (It is ICDAR code: read it first,
  then run it in a separate venv, never import it into ours.)
- Operating point: recall per defect type, false positives per board, clean-board flag rate.

## Milestones
- **M0 setup**: `prepare.py`, `selftest.py` (split sizes, no leakage, group split matches the
  table, labels round-trip, T channels match their source files), and time 1 epoch per arm.
- **M1 baseline**: A and T, 2 seeds each, on the full 900. Official-protocol and COCO numbers,
  seen vs unseen boards, per class.
- **M2 operating point + cost**: thresholds chosen on val only, then applied to test + clean.
- **M3 data curve**: A and T at 25/50/100/225/450/900.
- **M4 (optional) deploy**: TensorRT FP16 latency for A and T (reuse BDD `m4.py`). No INT8:
  BDD already showed TRT 10.16 INT8 is broken for YOLO11s on sm_120.

## Pre-registered tests (judged exactly as written, failures reported as failures)
1. **Unseen boards are harder**: arm A (mean over seeds) mAP50-95 on unseen boards is lower than
   on seen boards by more than the image-bootstrap 95% CI of the difference. Caveat to always
   print with it: the groups differ in layout, so "unseen" is also "different boards", not only
   "not trained on".
2. **The template helps**: T beats A on full-test mAP50-95, and the gap (T-A, mean over seeds)
   is larger than the larger seed spread of the two arms; otherwise the verdict is "tie", whatever
   the sign. Report the same comparison on unseen boards separately (descriptive).
3. **Fewer false alarms at the same recall**: on val, pick per arm the highest threshold that
   gives >= 99% recall (IoU 0.5, all classes pooled). Apply it to test. T has fewer false
   positives per board than A, and the bootstrap CI of the difference excludes 0. Clean-board
   flag rate is reported next to it but not judged, because T's clean input (template, template,
   0) is close to trivial, so a win there is expected and proves little.
4. **Cost of a miss**: for miss:false-alarm cost ratios 1, 10, 50, 200, choose the threshold that
   minimises expected cost on val and report the threshold, test recall, FP/board and cost per arm.
   Descriptive, no pass/fail.
5. **Labelled data needed**: report, per arm, the smallest training size that reaches 95% of that
   arm's own 900-image mAP50-95. Judged: T at 100 images beats A at 100 images by more than the
   seed spread at that size; otherwise "tie".

## v2: declared deviation (written 2026-10-21, before any v2 run)
- **Why**: v1 used the Ultralytics default `mosaic=1.0`. Post-hoc runs (`*_nomosaic`, seed 0)
  showed mosaic makes the model's boxes ~4 px inset per side versus the labels, which broke every
  strict-IoU number (A900 mAP50-95 17.4 -> 75.4, T900 7.7 -> 80.7) and made the data curve fall
  with more data. That was found AFTER the v1 results, so v2 is a deviation, not a fix of v1.
- **What changes**: `mosaic=0` for every run. Nothing else: same model, epochs (60), batch, seeds,
  close_mosaic setting, hsv off, splits, data-curve sizes and repetition, evaluators, bootstrap.
  Runs are named `<arm><n>_s<seed>_v2` (`train.py --recipe v2`). The two existing
  `A900_s0_nomosaic` / `T900_s0_nomosaic` runs have exactly the v2 recipe (args.yaml differs only
  in name), so they ARE the v2 seed-0 900 runs (linked, not retrained). The other v2 runs:
  `JOBS="A900_s1_v2 T900_s1_v2 A25_s0_v2 T25_s0_v2 A50_s0_v2 T50_s0_v2 A100_s0_v2 T100_s0_v2 A25_s1_v2 T25_s1_v2 A50_s1_v2 T50_s1_v2 A100_s1_v2 T100_s1_v2 A225_s0_v2 T225_s0_v2 A450_s0_v2 T450_s0_v2" make start`
- **Tests**: 1-5 judged exactly as written above, on the v2 runs (`compare.py --suffix _v2`,
  `operating.py --suffix _v2`; IoU 0.5 is the judged test 3/4, IoU 0.33 stays a labelled post-hoc
  sensitivity). Official-protocol port on the 900 runs.
- **v1 verdicts are kept** and reported first in the README (v1 -> mosaic finding -> v2). Neither
  set replaces the other. v1 outputs (`runs/compare.json`, `runs/operating*.json`) are not touched.
- Why mosaic causes the inset on this dataset stays unproven; v2 does not test that.

## Known traps
- **Most defects are artificial**: the authors drew extra defects onto each image ("around 3 to 12
  per image"). A model can learn their drawing style. Never claim the results transfer to a real
  line. Say "on DeepPCB".
- The images are already registered and binarised, so the hard parts of real AOI (alignment,
  lighting) are done for us. T's advantage depends on that alignment.
- Crops from the same board group may overlap or repeat patterns, so seen-board test scores can be
  inflated by near-duplicates. That is exactly why test 1 exists. Do not headline the seen-board
  number.
- The official metric uses IoU 0.33, which is looser than COCO's 0.5. Always say which one a
  number uses. 98.6% is IoU 0.33.
- Ceiling effect: if A is already ~99 at IoU 0.33, T cannot visibly win there. That is why the
  tests use mAP50-95 and operating-point false positives.
- `--limit`-style subsets must be hash-ordered, never "first N lines" (the lists are sorted by
  group, so the first 100 lines are all one board).

## Fixed before any trained result
`prepare.py` (both arms, hard links for A, PNG for T), `selftest.py` 22/22, `selftest_eval.py`
6/6 (our mAP50-95, mAP50 and AP@0.33 == pycocotools to 1e-6 on synthetic detections, full test
and unseen subset). Timing: ~10.5 s/epoch for A and T alike (4.1 GB VRAM). Fixed: **60 epochs**,
batch 16, `val=False` during training, close_mosaic 5, cos lr, COCO yolo11s (same file as BDD,
sha256 85a76fe8...). **Data-curve change (before any result): sizes 25/50/100/225/450/900, not
250/500**: each small subset is repeated to 900 entries with distinct hard-link names
(`data/yolo/<arm>/rep/`; Ultralytics' label cache would collapse a path listed twice), so every
run has the same 60 epochs and the same 3420 steps. Without this a 25-image run gets 36x fewer
steps (or, at 36x the epochs, ~45 min of per-epoch overhead). Labels on disk are
space-separated, not comma-separated as the DeepPCB README says (parser accepts both).
**Tests 3-4 matching, fixed before any result**: a defect is caught when any detection at or above
the threshold has IoU >= 0.5 with it, class ignored (class-agnostic; the board is pulled either
way); class-aware printed as a sensitivity (`operating.py`). Seeds averaged as in test 2.
**Official protocol**: DeepPCB's script.py cannot run as shipped (py2.7 + `Polygon`, and its
parser expects comma-separated 8-point lines with class names while gt.zip has space-separated
4-number lines with ids). `official_eval.py` is a py3 PORT of its logic (gt-first greedy matching,
IoU > 0.33, pooled uninterpolated AP, F at a val-chosen threshold + test-oracle F). Always call
it "our port of the official protocol", never "the official script".
