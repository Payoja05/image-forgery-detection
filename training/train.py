"""Train ForensicUNet on the synthetic tampering dataset.

    python -m training.train --epochs 12 --batch-size 8

Outputs
  checkpoints/forensic_unet_best.pt   best epoch by validation loss
  reports/training_history.json       per-epoch train/val loss and val metrics

Loss = BCE + soft Dice. BCE gives stable per-pixel gradients; Dice counters
the class imbalance (most pixels are authentic) and directly optimises
region overlap.

After training, the image-level decision threshold is calibrated on the
VALIDATION split (threshold that maximises image-level F1) and stored in the
checkpoint. The test split is never touched here - see evaluation/evaluate_model.py.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config.settings import (  # noqa: E402
    CHECKPOINT_PATH, HISTORY_PATH, IMAGE_SCORE_TOP_FRACTION, SEED, SYNTH_DIR, TRAIN_CROP,
)
from evaluation.metrics import best_f1_threshold, pixel_metrics, roc_auc  # noqa: E402
from inference.localization import image_level_score  # noqa: E402
from models.model_loader import get_device, save_checkpoint  # noqa: E402
from models.unet import ForensicUNet  # noqa: E402
from training.dataset import TamperDataset  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    p = torch.sigmoid(logits)
    inter = (p * target).sum(dim=(1, 2, 3))
    denom = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return (1 - (2 * inter + eps) / (denom + eps)).mean()


def loss_fn(logits, target):
    return F.binary_cross_entropy_with_logits(logits, target) + dice_loss(logits, target)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    losses, scores, labels, ious, f1s = [], [], [], [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        losses.append(loss_fn(logits, y).item() * len(x))
        prob = torch.sigmoid(logits).cpu().numpy()[:, 0]
        gt = y.cpu().numpy()[:, 0]
        for p, g in zip(prob, gt):
            scores.append(image_level_score(p, IMAGE_SCORE_TOP_FRACTION))
            labels.append(int(g.any()))
            if g.any():
                m = pixel_metrics(p >= 0.5, g > 0.5)
                ious.append(m["iou"])
                f1s.append(m["f1"])
    thr, img_f1 = best_f1_threshold(np.array(labels), np.array(scores))
    return {
        "loss": float(sum(losses) / len(loader.dataset)),
        "pixel_iou": float(np.mean(ious)) if ious else 0.0,
        "pixel_f1": float(np.mean(f1s)) if f1s else 0.0,
        "image_auc": roc_auc(np.array(labels), np.array(scores)),
        "image_f1_at_best_thr": img_f1,
        "image_threshold": thr,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=SYNTH_DIR)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--patience", type=int, default=4, help="early-stopping patience (epochs)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="debug: use only N training samples")
    args = ap.parse_args()

    set_seed(SEED)
    device = get_device()
    print(f"device={device}  threads={torch.get_num_threads()}")

    train_ds = TamperDataset(args.data / "train", crop=TRAIN_CROP, train=True)
    if args.limit:
        train_ds.records = train_ds.records[: args.limit]
    val_ds = TamperDataset(args.data / "val", crop=None, train=False)
    g = torch.Generator().manual_seed(SEED)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                          generator=g, drop_last=True, persistent_workers=args.workers > 0)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    print(f"train={len(train_ds)}  val={len(val_ds)}")

    model = ForensicUNet(pretrained_encoder=True).to(device)
    encoder = [p for n, p in model.named_parameters() if n.startswith(("conv1", "bn1", "layer"))]
    decoder = [p for n, p in model.named_parameters() if not n.startswith(("conv1", "bn1", "layer"))]
    opt = torch.optim.AdamW([{"params": encoder, "lr": args.lr * 0.5}, {"params": decoder, "lr": args.lr}],
                            weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    history = {"config": {**{k: str(v) for k, v in vars(args).items()}, "seed": SEED, "train_crop": TRAIN_CROP,
                          "device": str(device), "loss": "BCE + soft Dice"}, "epochs": []}
    best, bad_epochs = float("inf"), 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, run, n = time.time(), 0.0, 0
        for step, (x, y) in enumerate(train_dl, 1):
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            run += loss.item() * len(x)
            n += len(x)
            if step % 50 == 0:
                print(f"  epoch {epoch} step {step}/{len(train_dl)} loss {run / n:.4f}  ({time.time() - t0:.0f}s)", flush=True)
        sched.step()
        val = validate(model, val_dl, device)
        rec = {"epoch": epoch, "train_loss": run / n, **{f"val_{k}": v for k, v in val.items()},
               "seconds": round(time.time() - t0, 1)}
        history["epochs"].append(rec)
        print(json.dumps(rec), flush=True)

        if val["loss"] < best:
            best, bad_epochs = val["loss"], 0
            save_checkpoint(model, CHECKPOINT_PATH, {
                "arch_name": ForensicUNet.ARCH_NAME, "epoch": epoch,
                "image_threshold": val["image_threshold"],
                "image_threshold_source": "maximises image-level F1 on the validation split",
                "val_metrics": val, "train_crop": TRAIN_CROP,
                "trained_on": "Synthetic tampering dataset built from Oxford-IIIT Pet (see data/README.md)",
            })
            print(f"  saved new best checkpoint (val loss {best:.4f})")
        else:
            bad_epochs += 1
        history["best_epoch"] = min(history["epochs"], key=lambda r: r["val_loss"])["epoch"]
        HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        HISTORY_PATH.write_text(json.dumps(history, indent=2))
        if bad_epochs >= args.patience:
            print("early stopping")
            break
    print("done. best val loss", best)


if __name__ == "__main__":
    main()
