"""
train.py
--------
YOLOv8x fine-tuning entry point.

Can be called from a notebook or from the command line:

    python -m src.train \\
        --data    /content/rpc_subset/dataset.yaml \\
        --config  configs/train_config.yaml \\
        --project /content/runs \\
        --name    yolov8x_rpc

All hyper-parameters can be overridden via CLI flags or by editing
configs/train_config.yaml.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Default config (mirrors configs/train_config.yaml)
# ---------------------------------------------------------------------------
DEFAULTS: dict = {
    "weights":     "yolov8x.pt",
    "epochs":      50,
    "imgsz":       640,
    "batch":       16,
    "lr0":         1e-3,
    "lrf":         1e-2,
    "optimizer":   "AdamW",
    "cos_lr":      True,
    "patience":    15,
    "workers":     4,
    "device":      0,
    "amp":         True,
    "mosaic":      1.0,
    "mixup":       0.1,
    "copy_paste":  0.1,
    "degrees":     5.0,
    "translate":   0.1,
    "scale":       0.5,
    "hsv_h":       0.015,
    "hsv_s":       0.7,
    "hsv_v":       0.4,
    "save_period": 10,
    "project":     "runs",
    "name":        "yolov8x_rpc",
    "pretrained":  True,
    "verbose":     True,
}


def load_config(config_path: str | Path | None) -> dict:
    """Merge DEFAULTS with a YAML config file (config wins)."""
    cfg = DEFAULTS.copy()
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            overrides = yaml.safe_load(f) or {}
        cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg


def train(
    data:        str | Path,
    config_path: str | Path | None = None,
    **overrides,
) -> YOLO:
    """
    Fine-tune YOLOv8x.

    Parameters
    ----------
    data        : path to dataset YAML (required)
    config_path : path to train_config.yaml (optional)
    **overrides : any YOLO training kwarg overrides

    Returns
    -------
    Trained YOLO model (pointing to best.pt)
    """
    cfg = load_config(config_path)
    cfg.update(overrides)   # explicit overrides take highest priority

    weights = cfg.pop("weights", "yolov8x.pt")
    print(f"Loading base weights: {weights}")
    model = YOLO(weights)

    print("\n── Training configuration ──")
    for k, v in sorted(cfg.items()):
        print(f"  {k:20s}: {v}")
    print()

    results = model.train(
        data       = str(data),
        epochs     = cfg["epochs"],
        imgsz      = cfg["imgsz"],
        batch      = cfg["batch"],
        lr0        = cfg["lr0"],
        lrf        = cfg["lrf"],
        optimizer  = cfg["optimizer"],
        cos_lr     = cfg["cos_lr"],
        patience   = cfg["patience"],
        workers    = cfg["workers"],
        device     = cfg["device"],
        amp        = cfg["amp"],
        mosaic     = cfg["mosaic"],
        mixup      = cfg["mixup"],
        copy_paste = cfg["copy_paste"],
        degrees    = cfg["degrees"],
        translate  = cfg["translate"],
        scale      = cfg["scale"],
        hsv_h      = cfg["hsv_h"],
        hsv_s      = cfg["hsv_s"],
        hsv_v      = cfg["hsv_v"],
        save_period = cfg["save_period"],
        project    = cfg["project"],
        name       = cfg["name"],
        pretrained = cfg["pretrained"],
        verbose    = cfg["verbose"],
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    print(f"\nTraining complete. Best weights → {best}")

    # Load and return best model
    return YOLO(str(best))


def evaluate(
    model_path: str | Path,
    data:       str | Path,
    imgsz:      int  = 640,
    batch:      int  = 16,
    device:     int  = 0,
) -> dict:
    """
    Run validation and print metrics.

    Returns dict with map50, map, precision, recall.
    """
    model   = YOLO(str(model_path))
    metrics = model.val(
        data    = str(data),
        imgsz   = imgsz,
        batch   = batch,
        device  = device,
        verbose = True,
    )
    result = {
        "map50":      metrics.box.map50,
        "map":        metrics.box.map,
        "precision":  metrics.box.mp,
        "recall":     metrics.box.mr,
    }
    print("\n── Validation Metrics ──")
    for k, v in result.items():
        print(f"  {k:12s}: {v:.4f}")
    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YOLOv8x Retail Pricing — Training")
    p.add_argument("--data",    required=True,  help="Path to dataset YAML")
    p.add_argument("--config",  default=None,   help="Path to train_config.yaml")
    p.add_argument("--weights", default=None,   help="Base weights (default: yolov8x.pt)")
    p.add_argument("--epochs",  type=int,   default=None)
    p.add_argument("--batch",   type=int,   default=None)
    p.add_argument("--imgsz",   type=int,   default=None)
    p.add_argument("--device",  default=None)
    p.add_argument("--project", default=None)
    p.add_argument("--name",    default=None)
    return p.parse_args()


if __name__ == "__main__":
    args    = _parse_args()
    kwargs  = {k: v for k, v in vars(args).items()
               if k not in ("data", "config") and v is not None}
    train(data=args.data, config_path=args.config, **kwargs)
