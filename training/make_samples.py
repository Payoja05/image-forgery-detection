"""Copy a few labelled TEST-split images into assets/samples for the app gallery.

    python -m training.make_samples

Picks the first image of each manipulation type in file order (no
cherry-picking on model performance). Images are derived from the
Oxford-IIIT Pet Dataset (CC BY-SA 4.0).
"""
import csv
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config.settings import SAMPLES_DIR, SYNTH_DIR  # noqa: E402

TYPES = [("authentic", "Authentic", "no edit"), ("copy_move", "Manipulated", "Copy-Move"),
         ("splice_object", "Manipulated", "Splicing (object)"), ("splice_region", "Manipulated", "Splicing (region)")]


def main():
    test = SYNTH_DIR / "test"
    with open(test / "metadata.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for kind, label, type_label in TYPES:
        row = next(r for r in rows if r["type"] == kind)
        shutil.copy(test / "images" / row["file"], SAMPLES_DIR / row["file"])
        mask = f"{row['id']}_mask.png"
        shutil.copy(test / "masks" / f"{row['id']}.png", SAMPLES_DIR / mask)
        out.append({"file": row["file"], "mask": mask, "label": label, "type": kind, "type_label": type_label,
                    "source": "synthetic test split; derived from Oxford-IIIT Pet (CC BY-SA 4.0)"})
    (SAMPLES_DIR / "samples.json").write_text(json.dumps(out, indent=2))
    print(f"wrote {len(out)} samples to {SAMPLES_DIR}")


if __name__ == "__main__":
    main()
