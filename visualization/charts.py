"""Small matplotlib charts in the muted project palette."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from config.settings import PALETTE as P  # noqa: E402

BG = "#FFFDF8"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "axes.edgecolor": P["border"], "axes.labelcolor": P["ink"],
    "xtick.color": P["muted"], "ytick.color": P["muted"], "text.color": P["ink"], "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#EDE6DC",
    "grid.linewidth": 0.6, "legend.frameon": False,
})


def confusion_matrix_fig(c: dict):
    m = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    cmap = LinearSegmentedColormap.from_list("blues", ["#F4F8FC", "#9FC0DD", "#4F7CA8"])
    ax.imshow(m, cmap=cmap)
    ax.grid(False)
    for (i, j), v in np.ndenumerate(m):
        ax.text(j, i, str(v), ha="center", va="center", fontsize=12,
                color="white" if v > m.max() * 0.6 else P["ink"])
    ax.set_xticks([0, 1], ["Authentic", "Manipulated"])
    ax.set_yticks([0, 1], ["Authentic", "Manipulated"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    return fig


def roc_fig(roc: dict, auc: float | None):
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    ax.plot([0, 1], [0, 1], ls="--", color=P["border"], lw=1)
    ax.plot(roc["fpr"], roc["tpr"], color=P["blue_ink"], lw=2, label=f"model (AUC {auc:.3f})" if auc else "model")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def loss_curve_fig(epochs: list[dict]):
    e = [r["epoch"] for r in epochs]
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    ax.plot(e, [r["train_loss"] for r in epochs], color=P["blue_ink"], lw=2, marker="o", ms=3, label="Training loss")
    ax.plot(e, [r["val_loss"] for r in epochs], color=P["pink_ink"], lw=2, marker="o", ms=3, label="Validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCE + Dice loss")
    ax.legend()
    fig.tight_layout()
    return fig


def val_metrics_fig(epochs: list[dict]):
    e = [r["epoch"] for r in epochs]
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    ax.plot(e, [r["val_pixel_iou"] for r in epochs], color=P["sage_ink"], lw=2, marker="o", ms=3, label="Pixel IoU (val)")
    ax.plot(e, [r["val_pixel_f1"] for r in epochs], color=P["butter_ink"], lw=2, marker="o", ms=3, label="Pixel F1 (val)")
    ax.plot(e, [r["val_image_auc"] or 0 for r in epochs], color=P["blue_ink"], lw=2, marker="o", ms=3, label="Image ROC-AUC (val)")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Epoch")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def method_comparison_fig(names: list[str], ious: list[float], aucs: list[float]):
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.bar(x - 0.18, ious, 0.36, color=P["blue"], edgecolor=P["blue_ink"], label="Mean IoU")
    ax.bar(x + 0.18, aucs, 0.36, color=P["sage"], edgecolor=P["sage_ink"], label="Pixel ROC-AUC")
    for xi, (a, b) in enumerate(zip(ious, aucs)):
        ax.text(xi - 0.18, a + 0.02, f"{a:.2f}", ha="center", fontsize=8)
        ax.text(xi + 0.18, b + 0.02, f"{b:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x, names)
    ax.set_ylim(0, 1.1)
    ax.legend(loc="upper left", ncols=2)
    fig.tight_layout()
    return fig
