"""Train the SAR oil-spill segmenter.

    python scripts/train_oil.py --epochs 12 --batch-size 8

Adapted from SAMUDRA's train_segmenter.py. Two things differ here:

  * `import trident` first, which routes TLS through the OS trust store. The
    original run could not fetch the ImageNet backbone on this network and fell
    back to random initialisation, which the code itself calls materially
    worse. With the trust store in place the pretrained backbone downloads, and
    that is the single largest quality difference available.
  * The checkpoint lands in TRIDENT's weights directory so the API can serve it.

`representative` is written into the checkpoint and is true only for a full GPU
run. It travels with the weights and surfaces in the UI.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import trident  # noqa: F401  -- installs the OS trust store before torch fetches weights

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trident.sar.datasets import BinaryClassification, SOSSegmentation  # noqa: E402
from trident.sar.model import (  # noqa: E402
    DeepLabV3Plus,
    evaluate,
    evaluate_aux,
    pick_device,
    save_metrics,
    train_one_epoch,
)

SAMUDRA_DATA = Path.home() / "Desktop" / "DESKTOP_2" / "SIH_Demo" / "data" / "raw" / "oilspill"


def main() -> None:
    ap = argparse.ArgumentParser(description="Train DeepLabv3+ on the SAR oil-spill sets.")
    ap.add_argument("--sos", default=str(SAMUDRA_DATA / "sos"))
    ap.add_argument("--binary", default=str(SAMUDRA_DATA / "binary" / "data"))
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--aux-weight", type=float, default=0.4)
    ap.add_argument("--no-aux", action="store_true")
    ap.add_argument("--train-limit", type=int, default=None)
    ap.add_argument("--test-limit", type=int, default=None)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--amp", action="store_true", help="mixed precision, needed to fit 6 GB")
    ap.add_argument("--checkpoint", default="weights/oil_seg.pt")
    ap.add_argument("--metrics", default="artifacts/oil_metrics.json")
    ap.add_argument("--plot", default="artifacts/oil_metrics.png")
    a = ap.parse_args()

    device = pick_device(a.cpu)
    representative = device.type == "cuda" and not a.train_limit

    train_ds = SOSSegmentation(a.sos, "train", augment=True, limit=a.train_limit)
    test_ds = SOSSegmentation(a.sos, "test", augment=False, limit=a.test_limit)
    print(f"SOS train / test  : {len(train_ds)} / {len(test_ds)} image-mask pairs")

    aux_train = aux_test = None
    if not a.no_aux:
        lim = a.train_limit * 2 if a.train_limit else None
        aux_train = BinaryClassification(a.binary, split="train", augment=True, limit=lim)
        aux_test = BinaryClassification(a.binary, split="val", augment=False, limit=a.test_limit)
        print(f"binary aux train  : {len(aux_train)} images {aux_train.class_counts}")

    oil_frac = train_ds.oil_pixel_fraction(sample=min(300, len(train_ds)))
    oil_frac = float(min(max(oil_frac, 1e-3), 0.9))
    weights = torch.tensor([1.0, (1.0 - oil_frac) / oil_frac], dtype=torch.float32)
    weights = weights / weights.mean()
    print(f"oil pixel fraction: {oil_frac * 100:.2f}%   class weights "
          f"[bg {weights[0]:.3f}, oil {weights[1]:.3f}]")

    def loader(ds, shuffle):
        return DataLoader(
            ds, batch_size=a.batch_size, shuffle=shuffle, num_workers=a.workers,
            pin_memory=(device.type == "cuda"), drop_last=False,
            persistent_workers=a.workers > 0,
        )

    seg_train, seg_test = loader(train_ds, True), loader(test_ds, False)
    aux_train_dl = loader(aux_train, True) if aux_train else None
    aux_test_dl = loader(aux_test, False) if aux_test else None

    model = DeepLabV3Plus(num_classes=2, pretrained=not a.no_pretrained).to(device)
    print(f"model             : DeepLabv3+ / ResNet-50, "
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f} M parameters")
    print(f"pretrained        : {not a.no_pretrained}")

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(a.epochs, 1))

    history: list[dict] = []
    best_iou = -1.0
    ckpt = Path(a.checkpoint)
    ckpt.parent.mkdir(parents=True, exist_ok=True)

    for ep in range(1, a.epochs + 1):
        print(f"\n-- epoch {ep}/{a.epochs} " + "-" * 50, flush=True)
        tr = train_one_epoch(
            model, seg_train, opt, device, weights,
            aux_loader=aux_train_dl, aux_weight=a.aux_weight, log_every=100,
        )
        sched.step()
        ev = evaluate(model, seg_test, device)
        oil_iou = ev["per_class"]["oil"]["iou"]
        print(f"   seg loss {tr['seg_loss']:.4f}   aux loss {tr['aux_loss']:.4f}   "
              f"{tr['seconds']:.0f}s")
        print(f"   mean IoU {ev['mean_iou']:.4f}   oil IoU {oil_iou:.4f}   "
              f"oil recall {ev['per_class']['oil']['recall']:.4f}", flush=True)
        history.append({"epoch": ep, **tr, "mean_iou": ev["mean_iou"], "oil_iou": oil_iou})

        if oil_iou > best_iou:
            best_iou = oil_iou
            torch.save(
                {
                    "model": model.state_dict(),
                    "arch": "deeplabv3plus_resnet50",
                    "num_classes": 2,
                    "classes": ["background", "oil"],
                    "input_size": 256,
                    "normalisation": "imagenet",
                    "mask_threshold": 128,
                    "epoch": ep,
                    "oil_iou": oil_iou,
                    "representative": representative,
                    "pretrained_backbone": not a.no_pretrained,
                },
                ckpt,
            )
            print(f"   checkpoint saved (oil IoU {oil_iou:.4f})", flush=True)

    seg_metrics = evaluate(model, seg_test, device)
    aux_metrics = evaluate_aux(model, aux_test_dl, device) if aux_test_dl else None

    metrics = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "representative": representative,
        "note": (
            "Full GPU training run."
            if representative
            else "NOT REPRESENTATIVE: reduced subset and/or CPU run."
        ),
        "device": str(device),
        "platform": platform.platform(),
        "epochs": a.epochs,
        "batch_size": a.batch_size,
        "lr": a.lr,
        "pretrained_backbone": not a.no_pretrained,
        "train_pairs": len(train_ds),
        "test_pairs": len(test_ds),
        "oil_pixel_fraction": oil_frac,
        "class_weights": weights.tolist(),
        "segmentation": seg_metrics,
        "classification": aux_metrics,
        "history": history,
        "checkpoint": str(ckpt),
    }
    save_metrics(metrics, Path(a.metrics), Path(a.plot))

    print("\n" + "=" * 78)
    cm = np.array(seg_metrics["confusion_matrix"])
    print(f"  confusion matrix   [[{cm[0,0]:>12,} {cm[0,1]:>12,}]   true background")
    print(f"                      [{cm[1,0]:>12,} {cm[1,1]:>12,}]]  true oil")
    for name, m in seg_metrics["per_class"].items():
        print(f"  {name:<12} IoU {m['iou']:.4f}   precision {m['precision']:.4f}   "
              f"recall {m['recall']:.4f}")
    print(f"  mean IoU {seg_metrics['mean_iou']:.4f}   "
          f"pixel accuracy {seg_metrics['pixel_accuracy']:.4f}")
    for s, m in sorted(seg_metrics.get("per_sensor", {}).items()):
        print(f"  [{s:<8}] oil IoU {m['per_class']['oil']['iou']:.4f}   "
              f"recall {m['per_class']['oil']['recall']:.4f}")
    if aux_metrics:
        print(f"\n  aux accuracy {aux_metrics['pixel_accuracy']:.4f}")
    print(f"\n  weights -> {ckpt}")
    print("=" * 78)


if __name__ == "__main__":
    main()
