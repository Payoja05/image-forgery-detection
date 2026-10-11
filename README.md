# Image Forgery Detection and Tampering Localization

A computer-vision project that checks an image for signs of **local manipulation**
(splicing, copy-move) and **localizes** the suspicious pixels. It combines
classical image-forensic techniques (ELA, noise analysis, edge analysis, SIFT
copy-move detection) with a deep-learning segmentation network (U-Net with a
ResNet-18 encoder and an SRM noise stream) trained in this project, and presents
everything in a light, hand-made Streamlit interface.

> Every metric in this README and in the app was produced by running the code in
> this repository. No published benchmark numbers are presented as results.

---

## Contents
1. [Problem statement & objectives](#1-problem-statement--objectives)
2. [Features](#2-features)
3. [System architecture](#3-system-architecture)
4. [Project structure](#4-project-structure)
5. [Installation (Windows)](#5-installation-windows)
6. [Dataset preparation](#6-dataset-preparation)
7. [Model setup & training](#7-model-setup--training)
8. [Running the application](#8-running-the-application)
9. [Evaluation methodology](#9-evaluation-methodology)
10. [Experimental results](#10-experimental-results)
11. [Technical decisions](#11-technical-decisions)
12. [Testing](#12-testing)
13. [Limitations](#13-limitations)
14. [Future scope](#14-future-scope)
15. [Troubleshooting](#15-troubleshooting)

---

## 1. Problem statement & objectives

Image editors make it easy to paste an object from one photo into another
(*splicing*) or to clone a region within the same photo (*copy-move*). These edits
are often invisible, but they leave statistical traces: different JPEG
compression history, inconsistent sensor noise, unnatural boundaries or duplicated
texture.

**Goal:** given one image, (a) decide whether it shows signs of local manipulation
and (b) mark the manipulated pixels.

Objectives
- Image-level detection with a decision threshold calibrated on validation data.
- Pixel-level localization: probability heatmap → binary segmentation mask → overlay.
- Classical forensic maps for comparison and explanation.
- Honest evaluation on a held-out test split (image- and pixel-level metrics).
- A polished interface for uploading, analysing, comparing and exporting results.

## 2. Features

| Page | What it does |
|---|---|
| **Home** | Problem, objectives, feature cards, pipeline diagram, limitations, “start” button |
| **Upload & Analyse** | Drag-and-drop upload (JPG/JPEG/PNG ≤ 10 MB) with validation, preview, dimensions/size/format, image-quality warnings, 4 labelled example images with ground-truth masks, **batch analysis** with CSV export |
| **Image Forensics** | Error Level Analysis (adjustable JPEG quality), noise residual + noise-inconsistency map, SIFT copy-move detection (adjustable parameters), Canny edges + local-sharpness map — each with an explanation of what it cannot tell you |
| **Results & Visualizations** | Assessment, image-level score vs threshold, processing time, heatmap with legend, binary mask, overlay, **interactive segmentation threshold**, ground truth vs prediction (for examples), **classical vs deep-learning comparison**, downloads (annotated PNG, mask PNG, heatmap PNG, JSON report) |
| **Model Performance** | Test-set accuracy/precision/recall/F1/ROC-AUC, confusion matrix, ROC curve, IoU/Dice/pixel precision/recall, per-type results, ELA & noise baselines, GT vs predicted examples, training & validation curves — all read from generated JSON files |
| **About / Methodology** | Pipeline, dataset, architecture, training, evaluation, tools, limitations, future scope |

If the model weights are missing, the app says so clearly and explains how to
train them; the classical forensics page keeps working. It never shows a dummy
prediction.

## 3. System architecture

```
Input image (JPG/PNG)
   │  utils/image_validation.py     extension, real format, size ≤ 10 MB, corruption, min side 64 px, EXIF orientation
   ▼
Preprocessing                       preprocessing/image_preprocessing.py
   │  RGB uint8 kept untouched for display/forensics
   │  model input: resize long side ≤ 512 (INTER_AREA, never upscale) → reflect-pad to ×32 → ImageNet normalisation
   ▼
Traditional forensics               forensic_analysis/{ela,noise_analysis,edge_analysis,copy_move}.py
   ▼
Deep-learning model                 models/unet.py  (ForensicUNet, 14.4 M parameters)
   │  RGB ─┬────────────────────────────┐
   │       └─ SRM high-pass filters (3) ─┴─ concat (6 ch) → ResNet-18 encoder → U-Net decoder → 1 logit / pixel
   ▼
Tampering localization              inference/localization.py
   │  sigmoid → probability map (resized back to original size)
   │  mask = prob ≥ threshold → 3×3 opening → drop blobs < 0.1 % of image
   │  image score = mean of top 1 % pixel probabilities; suspicious if score ≥ calibrated threshold
   ▼
Visualization, export, evaluation   visualization/, utils/report_export.py, evaluation/
```

**Model input/output contract**

| | |
|---|---|
| Input | `1 × 3 × H × W` float32, RGB, `(x/255 − mean)/std` with ImageNet mean `(0.485,0.456,0.406)` / std `(0.229,0.224,0.225)`; `H, W` multiples of 32 |
| Output | `1 × 1 × H × W` logits; `sigmoid` → per-pixel probability that the pixel was tampered |
| Training input | random 256×256 crops of the 320×320 dataset images, horizontal flips |
| Inference input | whole image, long side ≤ 512 px |

## 4. Project structure

```
image-forgery-detection/
├── app.py                         Streamlit entry point + sidebar navigation
├── requirements.txt
├── .streamlit/config.toml         light theme (cream / blush palette, Patrick Hand headings)
├── config/settings.py             paths, model constants, palette
├── preprocessing/image_preprocessing.py
├── forensic_analysis/             ela.py · noise_analysis.py · edge_analysis.py · copy_move.py
├── models/                        resnet.py (ResNet-18 + official weights) · unet.py · model_loader.py
├── inference/                     localization.py · forgery_detector.py
├── training/                      build_synthetic_dataset.py · dataset.py · train.py · make_samples.py
├── evaluation/                    metrics.py · evaluate_model.py
├── visualization/                 overlays.py · charts.py
├── utils/                         image_validation.py · report_export.py
├── ui/                            theme.py · components.py · state.py
├── views/                         home · upload · forensics · results · performance · about
├── tests/                         pytest suite
├── checkpoints/                   trained weights (stored as 5 MB parts, rebuilt on first load)
├── reports/                       generated metrics, history, example predictions
├── assets/                        design reference + labelled example images
└── data/                          datasets (git-ignored, see data/README.md)
```

## 5. Installation (Windows)

Requires Python 3.10–3.12 (developed with 3.12.10). No GPU needed.

```powershell
cd image-forgery-detection
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

For an NVIDIA GPU, install the CUDA build of PyTorch from pytorch.org instead; the
code uses the GPU automatically when `torch.cuda.is_available()`.

**Why no torchvision / scikit-learn?** On the development laptop, Windows *Smart App
Control* blocked the unsigned compiled extensions of torchvision and SciPy
(`WinError 4551`). Rather than weakening a security setting, the project avoids
them: ResNet-18 is defined in `models/resnet.py` (parameter names identical to
torchvision, so the official ImageNet weights load directly, SHA-256 verified), and
ROC/AUC are implemented in NumPy in `evaluation/metrics.py` (unit-tested against
the Mann-Whitney statistic).

## 6. Dataset preparation

```powershell
python -m training.build_synthetic_dataset      # ~800 MB download + ~5 min generation
python -m training.make_samples                 # copies 4 labelled test images into assets/samples
```

The model is trained on a **synthetic tampering dataset generated by this
project** from the Oxford-IIIT Pet Dataset (CC BY-SA 4.0): object splicing using
the dataset's segmentation trimaps, random-region splicing and copy-move, with
pixel-exact masks. Source photos are split 70/15/15 % into train/val/test **before**
generation so no photo appears in two splits. 3,000 / 500 / 600 samples
(≈40 % authentic). Full details, and how to evaluate on CASIA v2 / Columbia /
CoMoFoD after a manual download, are in [`data/README.md`](data/README.md).

## 7. Model setup & training

The trained checkpoint is `checkpoints/forensic_unet_best.pt` (55 MB). In the GitHub repository it is
stored as twelve 5 MB parts (`forensic_unet_best.pt.part1` … `part12`) plus a SHA-256 file; the app joins
them automatically on first load and refuses a corrupt result. Nothing needs to be done by hand.

To reproduce the checkpoint:

```powershell
python -m training.train --epochs 10 --batch-size 8
```

* Encoder initialised from the official PyTorch ImageNet ResNet-18 weights
  (`resnet18-f37072fd.pth`, downloaded once to the torch cache and checksum-verified).
* Loss = BCE + soft Dice; AdamW (encoder lr 1.5e-4, decoder lr 3e-4, weight decay 1e-4);
  cosine schedule; seed 42.
* Validation after every epoch; the checkpoint with the lowest validation loss is kept;
  early stopping after 4 epochs without improvement.
* After each epoch the image-level threshold that maximises F1 on the **validation**
  split is computed and stored in the checkpoint (the test split is not used).
* History → `reports/training_history.json`.
* Approx. 15 min/epoch on an Intel Core Ultra 5 225H CPU (14 threads); 10 epochs ≈ 2.5 h.

Then evaluate on the held-out test split:

```powershell
python -m evaluation.evaluate_model
```

## 8. Running the application

```powershell
.venv\Scripts\activate
streamlit run app.py
```

Open http://localhost:8501. Typical flow: **Upload & Analyse** → pick an example or
upload a photo → **Analyse image** → **Results & Visualizations** (move the threshold
slider, download the report) → **Image Forensics** for the classical maps.

## 9. Evaluation methodology

`evaluation/evaluate_model.py` runs the model on every test image at its native size.

* **Image level.** Label = 1 if the ground-truth mask has any tampered pixel. Score =
  mean of the top 1 % pixel probabilities. Prediction = score ≥ threshold calibrated on
  the validation split. Reported: accuracy, precision, recall, F1, specificity,
  confusion matrix, ROC curve and ROC-AUC (threshold-free).
* **Pixel level** (manipulated images only, pixel threshold 0.5, after post-processing):
  IoU = TP/(TP+FP+FN), Dice = 2TP/(2TP+FP+FN), pixel precision = TP/(TP+FP), pixel
  recall = TP/(TP+FN); computed per image and averaged. Pixel ROC-AUC on 2,000 random
  pixels per image. Also the mean flagged area on authentic images (false positives).
* **Per manipulation type**: detection rate, mean IoU and Dice.
* **Classical baselines**: the smoothed ELA map and the noise-inconsistency map,
  binarised per image with Otsu's threshold, scored with the same pixel metrics.
* **Timing**: mean wall-clock inference time per image.

## 10. Experimental results

All numbers below were produced by `python -m training.train` and
`python -m evaluation.evaluate_model` on 2026-10-10 (CPU: Intel Core Ultra 5 225H) and are
stored in `reports/`. **Test set:** synthetic test split, 600 images
(232 authentic, 368 manipulated) built from source photos that
were never used in training or validation. The test split was used once, after training.

**Image-level detection** (threshold 0.780, chosen on the validation split)

| Accuracy | Precision | Recall | F1 | Specificity | ROC-AUC |
|---|---|---|---|---|---|
| 94.7 % | 95.9 % | 95.4 % | 95.6 % | 93.5 % | 0.983 |

Confusion matrix: TP 351, FN 17, FP 15, TN 217.

**Pixel-level localization** (mean over the 368 manipulated images, pixel threshold 0.5)

| IoU | Dice | Pixel precision | Pixel recall | Pixel ROC-AUC | False-positive area on authentic images |
|---|---|---|---|---|---|
| 0.840 | 0.883 | 0.910 | 0.876 | 0.994 | 0.3 % |

**Per manipulation type**

| Type | Images | Flagged suspicious | Mean IoU | Mean Dice |
|---|---|---|---|---|
| Splicing (object) | 165 | 100.0 % | 0.871 | 0.919 |
| Splicing (region) | 90 | 97.8 % | 0.891 | 0.921 |
| Copy-move | 113 | 86.7 % | 0.753 | 0.801 |
| Authentic | 232 | 6.5 % (false alarms) | – | – |

Copy-move is the hardest type: the copied pixels come from the same photo, so their
noise and compression statistics match the surroundings.

**Classical baselines on the same test images**

| Method | Mean IoU | Mean Dice | Pixel ROC-AUC |
|---|---|---|---|
| ELA, Otsu-binarised | 0.120 | 0.194 | 0.544 |
| Noise inconsistency, Otsu-binarised | 0.119 | 0.186 | 0.572 |
| ForensicUNet (this project) | 0.840 | 0.883 | 0.994 |

On this data the classical maps are close to chance (pixel ROC-AUC ≈ 0.55): they react to
edges and texture as much as to pasted regions. This compares methods on synthetic data
only and says nothing about real-world forgeries.

**Speed:** 149 ms mean model inference per 320×320 image on CPU.

**Training history** (best epoch = 10, by validation loss)

| Epoch | Train loss | Val loss | Val pixel IoU | Val image ROC-AUC |
|---|---|---|---|---|
| 1 | 1.1100 | 0.8515 | 0.668 | 0.958 |
| 2 | 0.8354 | 0.6923 | 0.734 | 0.951 |
| 3 | 0.7081 | 0.6339 | 0.747 | 0.951 |
| 4 | 0.6493 | 0.5574 | 0.803 | 0.972 |
| 5 | 0.6039 | 0.5536 | 0.791 | 0.960 |
| 6 | 0.5730 | 0.5253 | 0.825 | 0.974 |
| 7 | 0.5372 | 0.5057 | 0.843 | 0.975 |
| 8 | 0.5172 | 0.4963 | 0.856 | 0.978 |
| 9 | 0.5032 | 0.4933 | 0.859 | 0.978 |
| 10 | 0.4970 | 0.4898 | 0.863 | 0.980 |

**Important:** these results describe performance on synthetic forgeries similar to the
training data. They are **not** an estimate of accuracy on real-world manipulated images
(e.g. CASIA v2), which was not measured.

## 11. Technical decisions

* **Segmentation, not classification.** A classifier cannot output a mask, so a U-Net
  predicts per-pixel probabilities and the image-level decision is derived from them.
* **SRM noise stream.** Fixed rich-model high-pass filters (Fridrich & Kodovsky 2012)
  remove most image content and expose noise; RGB-N (Zhou et al., CVPR 2018) showed
  that they help tampering localization. Without them a network tends to learn
  *semantic* cues (“there is an animal here”).
* **Why not a published forensic model?** Candidates such as MVSS-Net or TruFor
  distribute weights via third-party drives or under non-commercial terms and target
  GPUs; their weights could not be verified here. A model trained by this repository
  is fully reproducible and its limits are known.
* **Why synthetic data?** Real forgery datasets require registration/manual download.
  Synthetic splicing with exact masks is a standard pre-training strategy; the
  evaluation script also accepts real datasets.
* **No destructive augmentation.** Only crops and flips — resizing, blurring or
  re-compressing during training would erase or fake the traces being learned.
* **Inference resolution.** Large images are down-scaled to 512 px (INTER_AREA) to keep
  CPU inference under a second; the mask is resized back to full resolution.

## 12. Testing

```powershell
python -m pytest -q tests
```

The suite (`tests/`) covers valid/invalid uploads (wrong extension, non-image bytes,
truncated JPEG, disguised BMP, tiny/empty files), preprocessing output sizes, ELA
output and its response to a region with different compression, noise-mismatch
detection, edge maps, copy-move detection on a cloned patch and no false alarm on
random texture, model loading errors (missing / corrupt / wrong architecture),
model input/output shapes, probability-map resolution, mask post-processing,
image-level score, overlay generation, JSON report export, and all metric formulas
(including ROC-AUC against the rank statistic).

Last run: **48 passed** (`python -m pytest -q tests`, 2026-10-10), including headless
Streamlit `AppTest` checks that render all six pages and drive the example → analyse →
results → threshold-slider flow with the trained model (`tests/test_app.py`).

## 13. Limitations

* Trained only on synthetic forgeries built from pet photos; performance on real-world
  manipulations (CASIA, social-media images) has **not** been measured and is expected
  to be considerably lower.
* Not designed for AI-generated images, diffusion inpainting, object removal, or global
  edits (filters, colour grading).
* Down-scaling to 512 px can erase traces of very small edits in large photos.
* The image score is a heuristic, not a calibrated probability of forgery.
* Copy-move detection reports only the largest geometrically consistent cluster and
  fails on flat or heavily transformed copies.
* ELA / noise / edge maps respond to image content as well as to tampering.

## 14. Future scope

* Fine-tune and evaluate on CASIA v2 / CoMoFoD / Columbia (script already supports them).
* Tile-based full-resolution inference.
* Add inpainting / object-removal samples to the synthetic generator.
* Stronger backbones (SegFormer, EfficientNet) on a GPU; learnable constrained convolutions.
* Fuse classical maps as extra network inputs.

## 15. Troubleshooting

| Problem | Fix |
|---|---|
| `Model weights not found` in the app | Run `python -m training.train` (after building the dataset) or copy `forensic_unet_best.pt` into `checkpoints/` |
| `DLL load failed … An Application Control policy has blocked this file` | Windows Smart App Control blocks an unsigned package binary. This project does not need torchvision/SciPy; uninstall them, or ask an administrator — do not disable security features casually |
| `python` opens the Microsoft Store | Python is not installed; install Python 3.12 from python.org (`winget install Python.Python.3.12`) |
| Dataset download fails | Check your internet connection; the script tries two official mirrors and verifies MD5 checksums. Delete partial files in `data/raw/oxford-iiit-pet/` and retry |
| Training is slow | Use `--limit 800 --epochs 3` for a quick run, or a CUDA GPU |
| Model Performance page says metrics unavailable | Run `python -m evaluation.evaluate_model` |
| Port 8501 in use | `streamlit run app.py --server.port 8502` |

## Credits

* Oxford-IIIT Pet Dataset — O. M. Parkhi, A. Vedaldi, A. Zisserman, C. V. Jawahar, *Cats and Dogs*, CVPR 2012 (CC BY-SA 4.0).
* ResNet — K. He et al., CVPR 2016; ImageNet weights published by the PyTorch team (BSD-3).
* U-Net — O. Ronneberger et al., MICCAI 2015.
* SRM filters — J. Fridrich, J. Kodovsky, IEEE TIFS 2012; RGB-N — P. Zhou et al., CVPR 2018.
* SIFT copy-move — I. Amerini et al., IEEE TIFS 2011.
* Patrick Hand font — Google Fonts (OFL).
