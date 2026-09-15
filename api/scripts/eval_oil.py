"""Score a trained oil checkpoint against held-out ground truth.

Reports IoU, precision and recall per sensor, and renders scene / truth /
prediction triplets. The visual panel exists because a single IoU number does
not tell you *how* a segmenter is wrong: over-prediction that floods the scene
and under-prediction that catches only the slick core can land on similar
scores while meaning opposite things operationally.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import trident  # noqa: F401

import numpy as np
from PIL import Image

from trident.sar.segment import SARSegmenter

SAMUDRA = Path.home() / "Desktop" / "DESKTOP_2" / "SIH_Demo" / "data" / "raw" / "oilspill"


def score(pred: np.ndarray, truth: np.ndarray) -> dict:
    tp = int((pred & truth).sum())
    fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum())
    tn = int((~pred & ~truth).sum())
    return {
        "iou": tp / max(tp + fp + fn, 1),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "accuracy": (tp + tn) / max(tp + fp + fn + tn, 1),
    }


def panel(scene: np.ndarray, truth: np.ndarray, pred: np.ndarray) -> Image.Image:
    """Scene, ground truth and prediction side by side."""
    h, w = truth.shape
    grey = np.stack([scene] * 3, axis=-1).astype(np.uint8)

    def tint(mask, colour):
        out = grey.copy()
        overlay = out.astype(np.float32)
        for c in range(3):
            overlay[..., c] = np.where(
                mask, 0.45 * overlay[..., c] + 0.55 * colour[c], overlay[..., c]
            )
        return overlay.astype(np.uint8)

    strip = np.concatenate(
        [grey, tint(truth, (60, 200, 120)), tint(pred, (240, 80, 80))], axis=1
    )
    return Image.fromarray(strip)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="weights/oil_seg.pt")
    ap.add_argument("--sos", default=str(SAMUDRA / "sos" / "test"))
    ap.add_argument("--limit", type=int, default=120)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--panels", type=int, default=6)
    ap.add_argument("--out", default="artifacts/oil_eval")
    a = ap.parse_args()

    seg = SARSegmenter(a.checkpoint)
    print(f"checkpoint        : {a.checkpoint}")
    print(f"representative    : {seg.meta.get('representative')}   "
          f"epoch {seg.meta.get('epoch')}   reported oil IoU {seg.meta.get('oil_iou')}")
    print(f"device            : {seg.device}")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    totals: dict[str, list[dict]] = {}
    saved = 0

    for sensor in ("palsar", "sentinel"):
        imgd = Path(a.sos) / sensor / "image"
        labd = Path(a.sos) / sensor / "label"
        if not imgd.is_dir():
            continue

        files = sorted(imgd.glob("*.png"))[: a.limit]
        rows = []
        for f in files:
            lab = labd / f.name
            if not lab.exists():
                continue
            scene = np.array(Image.open(f).convert("L"))
            truth = np.array(Image.open(lab).convert("L")) >= 128
            pred = seg.predict(scene, threshold=a.threshold).mask
            rows.append(score(pred, truth))

            if saved < a.panels and truth.mean() > 0.12:
                panel(scene, truth, pred).save(out / f"{sensor}-{f.stem}.png")
                saved += 1

        totals[sensor] = rows

    print("\n" + "=" * 70)
    print(f"{'sensor':<12}{'n':>5}{'IoU':>9}{'precision':>12}{'recall':>9}")
    all_rows: list[dict] = []
    for sensor, rows in totals.items():
        if not rows:
            continue
        all_rows += rows
        print(f"{sensor:<12}{len(rows):>5}"
              f"{np.mean([r['iou'] for r in rows]):>9.4f}"
              f"{np.mean([r['precision'] for r in rows]):>12.4f}"
              f"{np.mean([r['recall'] for r in rows]):>9.4f}")
    if all_rows:
        print("-" * 70)
        print(f"{'combined':<12}{len(all_rows):>5}"
              f"{np.mean([r['iou'] for r in all_rows]):>9.4f}"
              f"{np.mean([r['precision'] for r in all_rows]):>12.4f}"
              f"{np.mean([r['recall'] for r in all_rows]):>9.4f}")

    # A model that scores well on one band and badly on the other has learned a
    # sensor, not oil. Worth saying out loud rather than leaving in a table.
    if len(totals) == 2 and all(totals.values()):
        gap = abs(
            np.mean([r["iou"] for r in totals["palsar"]])
            - np.mean([r["iou"] for r in totals["sentinel"]])
        )
        if gap > 0.15:
            print(f"\n  !! {gap:.3f} IoU gap between L-band and C-band. That is a "
                  f"sensor-specific fit, not oil detection.")

    print(f"\n  panels -> {out}")
    print("=" * 70)


if __name__ == "__main__":
    main()
