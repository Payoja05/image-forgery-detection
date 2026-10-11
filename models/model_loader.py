"""Loading the trained ForensicUNet checkpoint with clear error messages."""
from __future__ import annotations

import hashlib
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


def assemble_from_parts(path: Path) -> bool:
    """Rebuild `<name>.pt` from `<name>.pt.part1..N` if only the parts exist.

    The checkpoint is stored in ~5 MB parts in the GitHub repository because
    single large uploads kept failing; `<name>.pt.sha256` guards the result.
    Returns True if the file was assembled.
    """
    parts = sorted(path.parent.glob(path.name + ".part*"), key=lambda p: int(p.suffix.removeprefix(".part")))
    if path.exists() or not parts:
        return False
    data = b"".join(p.read_bytes() for p in parts)
    digest_file = path.with_name(path.name + ".sha256")
    if digest_file.exists() and hashlib.sha256(data).hexdigest() != digest_file.read_text().strip():
        raise ModelLoadError(f"Checkpoint parts for '{path.name}' are incomplete or corrupt (SHA-256 mismatch).")
    path.write_bytes(data)
    return True


def load_model(path: str | Path = CHECKPOINT_PATH, device: torch.device | None = None) -> tuple[ForensicUNet, dict]:
    path = Path(path)
    assemble_from_parts(path)
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
