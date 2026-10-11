"""Build a synthetic tampering dataset with pixel-exact ground-truth masks.

Source photos: Oxford-IIIT Pet Dataset (Parkhi et al., 2012; CC BY-SA 4.0),
downloaded automatically from the official VGG site. Its foreground segmentation
trimaps give realistic *object-shaped* regions for splicing.

Manipulation types generated
  authentic      - processed exactly like the others, but nothing pasted
  splice_object  - an animal cut out of a donor photo (using its segmentation
                   mask) and pasted into the host photo
  splice_region  - a random smooth-shaped region from a donor photo
  copy_move      - a region of the host copied to another place in the same
                   photo (optionally scaled / rotated)

To create inconsistent forensic traces like real forgeries, donor content is
JPEG-compressed at a different quality and may be rescaled, blurred or
re-lit before pasting; edges are sometimes feathered. Every final image is
saved as JPEG (random quality) or PNG.

Leakage control: source photos are split into train/val/test FIRST; hosts and
donors for a sample are always drawn from the same split, so no source photo
(or variant of it) appears in more than one split.

Usage:  python -m training.build_synthetic_dataset  [--train 3000 --val 500 --test 600]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import random
import sys
import tarfile
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config.settings import RAW_DIR, SEED, SYNTH_DIR  # noqa: E402

SIZE = 320                  # stored image size (training takes random 256 crops)
TYPE_PROBS = {"authentic": 0.40, "splice_object": 0.25, "splice_region": 0.15, "copy_move": 0.20}


# ------------------------------------------------------------ source photos
PETS_FILES = {  # official VGG URLs + MD5 checksums (same as torchvision's OxfordIIITPet)
    "images.tar.gz": "5c4f3ee8e5d25df40f4fd59a7f44e54c",
    "annotations.tar.gz": "95a8c909bbe2e81eed6a22bccdf3f68f",
}
PETS_MIRRORS = ("https://thor.robots.ox.ac.uk/datasets/pets/", "https://www.robots.ox.ac.uk/~vgg/data/pets/data/")


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_pets() -> tuple[list[Path], Path]:
    base = RAW_DIR / "oxford-iiit-pet"
    base.mkdir(parents=True, exist_ok=True)
    for name, md5 in PETS_FILES.items():
        folder = base / name.split(".")[0]
        if folder.exists():
            continue
        archive = base / name
        if not archive.exists() or _md5(archive) != md5:
            for mirror in PETS_MIRRORS:
                try:
                    print(f"  downloading {mirror + name}")
                    urllib.request.urlretrieve(mirror + name, archive)
                    break
                except OSError as exc:
                    print(f"  failed: {exc}")
            if not archive.exists() or _md5(archive) != md5:
                raise RuntimeError(f"Could not download a verified copy of {name}")
        with tarfile.open(archive) as tar:
            tar.extractall(base, filter="data")       # 'data' filter blocks path traversal
        archive.unlink()
    images = sorted((base / "images").glob("*.jpg"))
    return images, base / "annotations" / "trimaps"


def read_rgb(path: Path) -> np.ndarray | None:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return None if img is None else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def fit_square(img: np.ndarray, rng: random.Random, mask: np.ndarray | None = None):
    """Resize so the short side is SIZE, then random-crop SIZE x SIZE."""
    h, w = img.shape[:2]
    s = SIZE / min(h, w)
    nw, nh = max(SIZE, round(w * s)), max(SIZE, round(h * s))
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    y, x = rng.randint(0, nh - SIZE), rng.randint(0, nw - SIZE)
    img = img[y:y + SIZE, x:x + SIZE]
    if mask is not None:
        mask = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)[y:y + SIZE, x:x + SIZE]
    return img, mask


# ------------------------------------------------------------ helpers
def jpeg(img: np.ndarray, q: int) -> np.ndarray:
    _, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def random_blob(h: int, w: int, rng: random.Random, area_range=(0.02, 0.20)) -> np.ndarray:
    """Smooth random closed shape covering roughly area_range of the image."""
    target = rng.uniform(*area_range) * h * w
    r = np.sqrt(target / np.pi)
    cx, cy = rng.uniform(r, w - r), rng.uniform(r, h - r)
    n = rng.randint(6, 12)
    angles = np.sort(np.array([rng.uniform(0, 2 * np.pi) for _ in range(n)]))
    radii = np.array([r * rng.uniform(0.6, 1.4) for _ in range(n)])
    pts = np.stack([cx + radii * np.cos(angles), cy + radii * np.sin(angles)], 1).astype(np.int32)
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [pts], 255)
    m = cv2.GaussianBlur(m, (0, 0), r * 0.15)
    return (m > 127).astype(np.uint8)


def perturb_donor(patch: np.ndarray, rng: random.Random) -> np.ndarray:
    """Give the donor content a different processing history."""
    patch = jpeg(patch, rng.randint(50, 95))
    if rng.random() < 0.3:
        patch = cv2.GaussianBlur(patch, (0, 0), rng.uniform(0.5, 1.2))
    if rng.random() < 0.3:
        alpha, beta = rng.uniform(0.85, 1.15), rng.uniform(-15, 15)
        patch = cv2.convertScaleAbs(patch, alpha=alpha, beta=beta)
    if rng.random() < 0.2:
        noise = np.random.default_rng(rng.randint(0, 2**31)).normal(0, rng.uniform(2, 6), patch.shape)
        patch = np.clip(patch.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return patch


def composite(host: np.ndarray, donor: np.ndarray, region: np.ndarray, rng: random.Random):
    """Paste donor pixels where region==1. Returns image and GT mask."""
    alpha = region.astype(np.float32)
    if rng.random() < 0.5:                                    # feathered edge
        alpha = cv2.GaussianBlur(alpha, (0, 0), rng.uniform(0.8, 2.0))
    a = alpha[..., None]
    out = (donor.astype(np.float32) * a + host.astype(np.float32) * (1 - a)).round().astype(np.uint8)
    return out, (alpha > 0.5).astype(np.uint8)


def place(obj_mask: np.ndarray, obj_img: np.ndarray, rng: random.Random, area_range=(0.03, 0.30)):
    """Rescale an object (mask + pixels) and put it at a random spot on a SIZE canvas."""
    ys, xs = np.nonzero(obj_mask)
    if len(ys) < 50:
        return None
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    m, p = obj_mask[y0:y1, x0:x1], obj_img[y0:y1, x0:x1]
    s = np.sqrt(rng.uniform(*area_range) * SIZE * SIZE / m.sum())
    nh, nw = int(m.shape[0] * s), int(m.shape[1] * s)
    if not (8 <= nh <= SIZE and 8 <= nw <= SIZE):
        return None
    m = cv2.resize(m, (nw, nh), interpolation=cv2.INTER_NEAREST)
    p = cv2.resize(p, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    y, x = rng.randint(0, SIZE - nh), rng.randint(0, SIZE - nw)
    cm = np.zeros((SIZE, SIZE), np.uint8)
    cp = np.zeros((SIZE, SIZE, 3), np.uint8)
    cm[y:y + nh, x:x + nw], cp[y:y + nh, x:x + nw] = m, p
    return cm, cp


# ------------------------------------------------------------ manipulations
def make_sample(kind, host, donor_path, trimap_dir, rng):
    if kind == "authentic":
        if rng.random() < 0.3:     # global edits do not count as local tampering
            host = cv2.convertScaleAbs(host, alpha=rng.uniform(0.9, 1.1), beta=rng.uniform(-10, 10))
        return host, np.zeros(host.shape[:2], np.uint8)

    if kind == "splice_object":
        donor = read_rgb(donor_path)
        tri = cv2.imread(str(trimap_dir / (donor_path.stem + ".png")), cv2.IMREAD_GRAYSCALE)
        if donor is None or tri is None:
            return None
        fg = np.isin(tri, (1, 3)).astype(np.uint8)          # 1 = pet, 3 = boundary
        placed = place(fg, perturb_donor(donor, rng), rng)
        if placed is None:
            return None
        region, donor_canvas = placed
        return composite(host, donor_canvas, region, rng)

    if kind == "splice_region":
        donor = read_rgb(donor_path)
        if donor is None:
            return None
        donor, _ = fit_square(donor, rng)
        region = random_blob(SIZE, SIZE, rng)
        return composite(host, perturb_donor(donor, rng), region, rng)

    if kind == "copy_move":
        src = random_blob(SIZE, SIZE, rng, (0.02, 0.12))
        ys, xs = np.nonzero(src)
        bh, bw = np.ptp(ys) + 1, np.ptp(xs) + 1
        angle = rng.uniform(-20, 20) if rng.random() < 0.4 else 0.0
        scale = rng.uniform(0.85, 1.15) if rng.random() < 0.4 else 1.0
        for _ in range(20):        # find a shift that keeps the copy inside and mostly non-overlapping
            dx, dy = rng.randint(-SIZE // 2, SIZE // 2), rng.randint(-SIZE // 2, SIZE // 2)
            if abs(dx) < bw * 0.8 and abs(dy) < bh * 0.8:
                continue
            cx, cy = xs.mean(), ys.mean()
            M = cv2.getRotationMatrix2D((cx, cy), angle, scale)
            M[:, 2] += (dx, dy)
            moved_mask = cv2.warpAffine(src, M, (SIZE, SIZE), flags=cv2.INTER_NEAREST)
            if moved_mask.sum() < 0.9 * src.sum() * scale * scale:
                continue           # fell off the canvas
            moved_img = cv2.warpAffine(host, M, (SIZE, SIZE), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            if rng.random() < 0.3:
                moved_img = perturb_donor(moved_img, rng)
            return composite(host, moved_img, moved_mask, rng)
        return None
    raise ValueError(kind)


def save(img, mask, out_dir: Path, sid: str, rng):
    if rng.random() < 0.8:
        q = rng.randint(70, 98)
        name = f"{sid}.jpg"
        cv2.imwrite(str(out_dir / "images" / name), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])
    else:
        q = None
        name = f"{sid}.png"
        cv2.imwrite(str(out_dir / "images" / name), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out_dir / "masks" / f"{sid}.png"), mask * 255)
    return name, q


def build_split(name, n, sources, trimap_dir, rng):
    out = SYNTH_DIR / name
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    kinds, weights = zip(*TYPE_PROBS.items())
    rows, i = [], 0
    while len(rows) < n:
        i += 1
        kind = rng.choices(kinds, weights)[0]
        host_path = rng.choice(sources)
        donor_path = rng.choice(sources)
        if donor_path == host_path and kind.startswith("splice"):
            continue
        host = read_rgb(host_path)
        if host is None:
            continue
        host, _ = fit_square(host, rng)
        res = make_sample(kind, host, donor_path, trimap_dir, rng)
        if res is None:
            continue
        img, mask = res
        frac = float(mask.mean())
        if kind != "authentic" and not (0.01 <= frac <= 0.45):
            continue
        sid = f"{name}_{len(rows):05d}"
        fname, q = save(img, mask, out, sid, rng)
        rows.append({
            "id": sid, "file": fname, "type": kind, "label": int(kind != "authentic"),
            "host": host_path.stem, "donor": donor_path.stem if kind.startswith("splice") else "",
            "final_jpeg_quality": q if q is not None else "png", "tampered_fraction": round(frac, 4),
        })
        if len(rows) % 250 == 0:
            print(f"  {name}: {len(rows)}/{n}")
    with open(out / "metadata.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{name}: wrote {len(rows)} samples to {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=3000)
    ap.add_argument("--val", type=int, default=500)
    ap.add_argument("--test", type=int, default=600)
    args = ap.parse_args()

    rng = random.Random(SEED)
    np.random.seed(SEED)
    print("Downloading / locating Oxford-IIIT Pet ...")
    images, trimap_dir = download_pets()
    print(f"{len(images)} source photos found")

    # split SOURCE photos first -> no source photo shared between splits
    images = images[:]
    rng.shuffle(images)
    n = len(images)
    splits = {"train": images[: int(0.7 * n)], "val": images[int(0.7 * n): int(0.85 * n)], "test": images[int(0.85 * n):]}
    for name, srcs in splits.items():
        (SYNTH_DIR / name).mkdir(parents=True, exist_ok=True)
        (SYNTH_DIR / name / "source_photos.txt").write_text("\n".join(p.stem for p in srcs))
        build_split(name, getattr(args, name), srcs, trimap_dir, rng)


if __name__ == "__main__":
    main()
