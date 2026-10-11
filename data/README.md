# Data

Datasets are **not** part of the source code. Everything in this folder except
this README is git-ignored.

```
data/
  raw/oxford-iiit-pet/        downloaded source photos + trimaps (~800 MB)
  synthetic/
    train/  images/  masks/  metadata.csv  source_photos.txt   (3000 samples)
    val/    ...                                                 (500 samples)
    test/   ...                                                 (600 samples)
```

## 1. Synthetic tampering dataset (used for training and evaluation)

```bash
python -m training.build_synthetic_dataset            # defaults: --train 3000 --val 500 --test 600
```

* Downloads the **Oxford-IIIT Pet Dataset** (Parkhi et al., CVPR 2012) from the
  official VGG server — `images.tar.gz` (~755 MB) and `annotations.tar.gz` (~19 MB) —
  and verifies the MD5 checksums. Licence: **CC BY-SA 4.0**.
* Generates four kinds of samples at 320×320 px:

| type | what happens | mask |
|---|---|---|
| `authentic` | resized/cropped (sometimes a global brightness change) | all zero |
| `splice_object` | an animal is cut out of a donor photo with its segmentation trimap and pasted | pasted animal |
| `splice_region` | a random smooth-shaped region of a donor photo is pasted | pasted region |
| `copy_move` | a region of the same photo is copied elsewhere (optional ±20° rotation, 0.85–1.15× scale) | the copy only |

* Donor content is JPEG-compressed at a different quality (50–95) and may be blurred,
  re-lit or noised; edges are feathered in half the cases. The final image is saved as
  JPEG (quality 70–98, 80 %) or PNG (20 %).
* **No leakage:** the 7,390 source photos are shuffled (seed 42) and split 70/15/15 %
  *before* any sample is generated. Hosts and donors of a sample always come from the
  same split. `source_photos.txt` lists the photos of each split.
* `metadata.csv` columns: `id, file, type, label, host, donor, final_jpeg_quality, tampered_fraction`.

## 2. Evaluating on a real forgery dataset (optional, manual download)

Real datasets need registration / manual download, so they are not fetched automatically.
Any dataset can be evaluated once it is arranged like this:

```
data/<name>/images/<id>.<jpg|png|tif>
data/<name>/masks/<id>.png         # white = tampered; omit (or all-black) for authentic images
```

```bash
python -m evaluation.evaluate_model --split-dir data/<name> --name "<Dataset name>" --out reports/<name>_metrics.json
```

(The app's Model Performance page reads `reports/test_metrics.json`; use a different
`--out` if you want to keep the synthetic results there.)

| dataset | content | masks | where |
|---|---|---|---|
| **CASIA v2.0** | 7,491 authentic + 5,123 tampered (splicing, copy-move), JPEG/TIFF | not in the original release; community ground truth at `github.com/namtpham/casia2groundtruth` | Kaggle mirrors (login required) |
| **Columbia Uncompressed Splicing** | 183 authentic + 180 spliced TIFF | edge masks provided | Columbia DVMM lab website (request form) |
| **CoMoFoD** | 260 copy-move sets with post-processing variants | yes | CoMoFoD website (registration) |

Notes for CASIA v2: tampered files are named `Tp_*`, authentic `Au_*`; rename the masks so
each mask has the same stem as its image (e.g. `Tp_D_CND_M_N_ani00018_sec00096_00138.png`)
and place authentic images in `images/` without a mask. Do not mix images derived from the
same source photo across training and test splits if you fine-tune on CASIA.
