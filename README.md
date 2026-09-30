# Camera-Based Conveyor Inspection & Tracking System

A classical computer-vision implementation of the five-module brief. Uses OpenCV, NumPy, scikit-image, and scikit-learn; no pretrained deep model.

## Setup

Run from this directory in Python 3.10 or newer:

```text
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python tools/make_test_video.py --output data/smoke.mp4
```

Use real footage for the final submission; the generated clip is a synthetic smoke fixture. A fixed phone camera pointed at a tabletop works as a conveyor stand-in.

**Calibration capture.** Tilt matters: head-on views leave the focal length under-determined, so tilt the board 20–45° and rotate it across views. `calibrate.py` prints the spread and warns below 10°. The plane homography is fitted to the **first** view only, so that view defines world scale — capture at least one board image lying flat on the belt.

## What each module demonstrates

| Module | Demonstrates |
|---|---|
| **A** Segmentation | Four segmentation families run on the same frames, compared on time and box count. Touching boxes are the discriminator: Felzenszwalb separates them (10/10 and 15/15 at IoU ≥ 0.5) because the concave step between two boxes is a low-gradient edge its region-adjacency minimum-cut refuses to merge, while watershed and Otsu both return the merged blob (0/10). Felzenszwalb is not used downstream because it costs ~100× more and over-segments the belt. |
| **B** Calibration | A real `calibrateCamera` intrinsic/distortion solve, then four camera models (orthographic, weak-perspective, affine, full-perspective) measured against a known box. Given `--true-size`, the tool ranks all four by error: on the fixture orthographic wins at 8.33% mean error while affine loses 43%. Which model wins is a property of the capture — wide FOV and close camera favour full-perspective, long focal length favours orthographic — so the tool reports the ranking instead of asserting a fixed order. |
| **C** Motion | A hand-built dense Lucas–Kanade that solves the 2×2 system itself, versus OpenCV's sparse feature flow. They disagree exactly where theory says they must: on the textureless face of a box the aperture problem leaves the spatial gradient matrix rank-deficient, so dense vectors are unreliable where sparse corners are absent entirely. A 6-parameter affine fit with RANSAC then keeps the coherent planar motion and rejects the inconsistent vectors, which is what makes the speed figure trustworthy. |
| **D** Tracking | A constant-velocity Kalman filter run through an explicit `predict → measure → update` cycle printed for 12 consecutive frames, then through a 4-frame simulated dropout where the state coasts on prediction alone. Measurement noise is representable via `--measurement-noise`, because on noise-free input the posterior equals the measurement and the filter provably does nothing. |
| **E** Recognition | Alignment, PCA-eigenbox, and Hu-moment classifiers scored on a held-out split, then re-scored under 15° rotation *and* relighting. The two perturbations behave differently because the shared preprocessing calls `equalizeHist`: relighting moves raw pixels by 49.4 grey levels but the prepared vector by only 0.007, while rotation still moves it by 0.149. So apparent lighting robustness belongs to the preprocessing, not the classifiers, and the residual spread is almost entirely rotation. |

## Module commands

### A — Segmentation

```text
python module_a_segmentation/run_segmentation.py --video data/conveyor.mp4 --output outputs/segmentation --frames 5
python tools/make_test_video.py --output data/smoke_touching.mp4 --layout touching
```

Splits-and-merges, watershed, Felzenszwalb, and mean shift on the same frames, writing a comparison grid, `segmentation_metrics.json`, and `segmentation_notes.txt`. Mean shift is comparison-only: it hits the box cap on every frame at roughly 250× the cost of watershed. The notes name **Otsu + external contours** as the downstream detector — watershed is compared but not called.

### B — Calibration

`--cols` and `--rows` count inner corners, not squares.

```text
python module_b_calibration/calibrate.py --images data/calibration_images --cols 9 --rows 6 --square-size 2.5 --output outputs/calibration.npz
python module_b_calibration/measure_box.py --calibration outputs/calibration.npz --reference-box 100,100,120,80 --reference-size 12 8 --unknown-box 300,120,180,100 --image data/conveyor.mp4
```

Add `--true-size 18 10` to score all four models against ground truth and print a ranking. Boxes are `x,y,width,height` in pixels.

### C — Optical flow

```text
python module_c_motion/optical_flow.py --video data/conveyor.mp4 --frame 0 --calibration outputs/calibration.npz --output outputs/optical_flow
python module_c_motion/optical_flow.py --video data/smoke.mp4 --cm-per-pixel 0.05
```

Coarse-to-fine pyramid, so it stays accurate past one window width where a single-level solve collapses. Speed is printed in cm/s when calibration or `--cm-per-pixel` is supplied. Measured 6.032 cm/s against a true 6.000.

### D — Kalman tracking

```text
python module_d_tracking/kalman_tracker.py --video data/conveyor.mp4 --output outputs/kalman --belt-velocity 120 0 --measurement-noise 1.5 --seed 0
```

`--cycle-frames` controls how many consecutive frames print the full cycle (default 12, brief asks ≥ 10). `--measurement-noise` adds seeded Gaussian noise to the detections so the filter has something to reject. On the 45-frame smoke clip (true motion 120 px/s) whole-run RMSE drops from 18.9 px to 0.13 px and 23.4 px to 0.50 px.

### E — Recognition

Crops go in `data/recognition/box_a/*.png`, `box_b/*.png`, and so on — three or more classes, ~10 crops each.

```text
python module_e_recognition/recognize.py --data-root data/recognition --output outputs/recognition
```

Held-out accuracy for all three classifiers plus an ensemble that majority-votes with the rotation-robust alignment classifier breaking ties. Measured on the bundled fixtures: 91.7% upright, 75.0% at 15° rotation, 91.7% relit.

## Integrated pipeline

```text
python pipeline.py --video data/conveyor.mp4 --calibration outputs/calibration.npz --recognition-root data/recognition --output outputs/pipeline
```

Writes `annotated.mp4`, `pipeline_results.csv`, and `pipeline_summary.json`, printing per track: ID, size in cm, speed in cm/s, and recognised type. Runs without calibration or recognition data too, reporting sizes unavailable and labels `unclassified`. Static detections are dropped because Otsu accepts belt strips and shadows as boxes; `--keep-static` disables that.

The same pipeline is importable — `run_pipeline(PipelineConfig(...), on_progress=...)` — which is what the UI uses.

## Web UI

```text
python app.py            # http://127.0.0.1:8000
```

Zero new dependencies: stdlib `http.server` plus the modules above, with a single-page dashboard. Use **Choose a video to upload** to drop in an MP4/MOV/AVI/MKV/WEBM (up to 512 MB) — it is written to `data/uploads/` and the path is filled in for you, so you do not have to type paths. Then run the pipeline with a live annotated preview and progress bar, and inspect KPI tiles, the per-track record table, belt-speed and trajectory charts, and the artifacts each module wrote.

Uploads are only accepted from the local UI, and the client-supplied filename is flattened to a safe stem inside `data/uploads/`, so a crafted name cannot write anywhere else. `--port` and `--host` are available; artifacts are served with byte-range support so the recorded video can seek.

> GitHub Pages cannot run this. It serves static files only, so it has no Python or OpenCV and the API would 404. Run it locally as above; the repository is the backup and the submission artefact.

## Verification

```text
python -m pytest
python -m ruff check .
python -m compileall common module_a_segmentation module_b_calibration module_c_motion module_d_tracking module_e_recognition pipeline.py app.py
```

51 tests pass. See `WORK_LOG_2.txt` for the measured comparisons behind the claims above, including the ones that came out unfavourably.