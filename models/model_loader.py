"""Loading the trained ForensicUNet checkpoint with clear error messages."""
from __future__ import annotations

from pathlib import Path

import torch

from config.settings import CHECKPOINT_PATH
from models.unet import ForensicUNet


class ModelLoadError(RuntimeError):
    """Raised when the checkpoint is missing or incompatible."""


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(model: ForensicUNet, path: Path, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"arch": ForensicUNet.ARCH_NAME, "state_dict": model.state_dict(), "meta": meta}, path)


def load_model(path: str | Path = CHECKPOINT_PATH, device: torch.device | None = None) -> tuple[ForensicUNet, dict]:
    path = Path(path)
    if not path.exists():
        raise ModelLoadError(
            f"Model weights not found at '{path}'. Train the model with "
            "`python -m training.train` (see README, section 'Model setup')."
        )
    device = device or get_device()
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except Exception as exc:  # corrupt / wrong file type
        raise ModelLoadError(f"Could not read checkpoint '{path.name}': {exc}") from exc

    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise ModelLoadError(f"'{path.name}' is not a ForensicUNet checkpoint (no state_dict).")
    if ckpt.get("arch") != ForensicUNet.ARCH_NAME:
        raise ModelLoadError(f"Checkpoint architecture '{ckpt.get('arch')}' != expected '{ForensicUNet.ARCH_NAME}'.")

    model = ForensicUNet(pretrained_encoder=False)
    try:
        model.load_state_dict(ckpt["state_dict"], strict=True)
    except RuntimeError as exc:
        raise ModelLoadError(f"Checkpoint weights do not match the model definition: {exc}") from exc
    model.to(device).eval()
    return model, ckpt.get("meta", {})
