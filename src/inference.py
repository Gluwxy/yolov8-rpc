"""
inference.py
------------
Image and video inference with product count + price overlay.

Usage:
    detector = ProductDetector(
        model_path    = "runs/yolov8x_rpc/weights/best.pt",
        catalog_path  = "rpc_subset/price_catalog.json",
    )
    # Image
    annotated, receipt = detector.infer_image("shelf.jpg")

    # Video
    totals = detector.process_video(
        video_path  = "checkout.mp4",
        output_path = "checkout_out.mp4",
    )
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from .pricing import PriceCatalog


# ---------------------------------------------------------------------------
# Colour palette  (BGR)
# ---------------------------------------------------------------------------
_GREEN  = (0, 220, 0)
_BLACK  = (0, 0, 0)
_WHITE  = (255, 255, 255)
_CYAN   = (200, 255, 0)
_PANEL  = (20, 20, 20)
_FONT   = cv2.FONT_HERSHEY_SIMPLEX


class ProductDetector:
    """
    Wraps a trained YOLOv8 model for retail product detection + pricing.

    Parameters
    ----------
    model_path   : path to best.pt (or any YOLO-compatible weights)
    catalog_path : path to price_catalog.json produced by RPCDatasetConverter
    conf         : default confidence threshold
    iou          : default NMS IoU threshold
    """

    def __init__(
        self,
        model_path:   str | Path,
        catalog_path: str | Path,
        conf: float = 0.30,
        iou:  float = 0.45,
    ):
        from ultralytics import YOLO
        self.model   = YOLO(str(model_path))
        self.catalog = PriceCatalog.from_json(catalog_path)
        self.conf    = conf
        self.iou     = iou

    # ------------------------------------------------------------------
    # Image inference
    # ------------------------------------------------------------------

    def infer_image(
        self,
        image: str | Path | np.ndarray,
        conf:  float | None = None,
        iou:   float | None = None,
    ) -> tuple[np.ndarray, dict]:
        """
        Run inference on a single image.

        Returns
        -------
        annotated : RGB numpy array with boxes + price labels + receipt bar
        receipt   : {"lines": [(name, count, subtotal)], "total": float,
                     "counts": {class_idx: count}}
        """
        conf = conf or self.conf
        iou  = iou  or self.iou

        if isinstance(image, (str, Path)):
            frame = cv2.imread(str(image))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame = image.copy()

        results = self.model(frame, conf=conf, iou=iou, verbose=False)[0]
        counts: dict[int, int] = defaultdict(int)

        for box in results.boxes:
            cls  = int(box.cls[0])
            cf   = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            name  = self.catalog.get_name(cls)
            price = self.catalog.get_price(cls)
            counts[cls] += 1

            cv2.rectangle(frame, (x1, y1), (x2, y2), _GREEN, 2)
            label = f"{name[:14]} | ¥{price:.1f}  {cf:.2f}"
            _put_label(frame, label, x1, y1)

        lines, total = self.catalog.compute_receipt(dict(counts))
        _draw_receipt_bar(frame, len(counts), total)

        return frame, {"lines": lines, "total": total, "counts": dict(counts)}

    # ------------------------------------------------------------------
    # Video inference
    # ------------------------------------------------------------------

    def process_video(
        self,
        video_path:  str | Path,
        output_path: str | Path,
        conf:        float | None = None,
        iou:         float | None = None,
        skip_frames: int = 1,
    ) -> dict:
        """
        Process a video, writing an annotated copy to output_path.

        Parameters
        ----------
        skip_frames : process every Nth frame (1 = every frame).
                      Higher values = faster processing, less accuracy.

        Returns
        -------
        dict with session "counts" {class_idx: count} and "total" float.
        """
        conf = conf or self.conf
        iou  = iou  or self.iou

        video_path  = Path(video_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps, (W, H),
        )

        session_counts: dict[int, int] = defaultdict(int)
        session_total  = 0.0
        last_counts: dict[int, int] = {}
        frame_idx = 0

        print(f"Processing {total_frames} frames …")
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % skip_frames == 0:
                    results = self.model(
                        frame, conf=conf, iou=iou, verbose=False
                    )[0]
                    frame_counts: dict[int, int] = defaultdict(int)

                    for box in results.boxes:
                        cls  = int(box.cls[0])
                        cf   = float(box.conf[0])
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        name  = self.catalog.get_name(cls)
                        price = self.catalog.get_price(cls)
                        frame_counts[cls] += 1

                        cv2.rectangle(frame, (x1, y1), (x2, y2), _GREEN, 2)
                        label = f"{name[:14]} | ¥{price:.1f}"
                        _put_label(frame, label, x1, y1)

                    # Update session (running max per class)
                    for cls, cnt in frame_counts.items():
                        if cnt > session_counts[cls]:
                            delta = cnt - session_counts[cls]
                            session_total     += delta * self.catalog.get_price(cls)
                            session_counts[cls] = cnt

                    last_counts = dict(frame_counts)

                # HUD
                frame_total = sum(
                    cnt * self.catalog.get_price(cls)
                    for cls, cnt in last_counts.items()
                )
                _draw_hud(frame, last_counts, frame_total, session_total)

                writer.write(frame)
                frame_idx += 1

                if frame_idx % 100 == 0:
                    print(f"  frame {frame_idx}/{total_frames}")
        finally:
            cap.release()
            writer.release()

        print(f"\nVideo saved → {output_path}")
        lines, total = self.catalog.compute_receipt(dict(session_counts))
        print("\n── Session Receipt ──")
        for name, cnt, sub in lines:
            print(f"  {name:30s} x{cnt:3d}  ¥{sub:.2f}")
        print(f"  {'TOTAL':30s}     ¥{total:.2f}")

        return {"counts": dict(session_counts), "total": total}


# ---------------------------------------------------------------------------
# Drawing utilities
# ---------------------------------------------------------------------------

def _put_label(frame: np.ndarray, label: str, x1: int, y1: int) -> None:
    (tw, th), _ = cv2.getTextSize(label, _FONT, 0.48, 1)
    y0 = max(0, y1 - th - 6)
    cv2.rectangle(frame, (x1, y0), (x1 + tw + 4, y1), (0, 90, 0), -1)
    cv2.putText(frame, label, (x1 + 2, y1 - 3), _FONT, 0.48, _WHITE, 1)


def _draw_receipt_bar(frame: np.ndarray, n_items: int, total: float) -> None:
    H, W = frame.shape[:2]
    cv2.rectangle(frame, (0, H - 38), (W, H), _BLACK, -1)
    text = f"Items: {n_items}   Total: ¥{total:.2f}"
    cv2.putText(frame, text, (12, H - 10), _FONT, 0.85, _CYAN, 2)


def _draw_hud(
    frame: np.ndarray,
    frame_counts: dict[int, int],
    frame_total: float,
    session_total: float,
) -> None:
    H, W = frame.shape[:2]
    n = sum(frame_counts.values())
    lines = [
        f"Frame items : {n}",
        f"Frame total : ¥{frame_total:.2f}",
        f"Session tot : ¥{session_total:.2f}",
    ]
    pw = 275
    ph = len(lines) * 28 + 12
    cv2.rectangle(frame, (W - pw - 10, 8), (W - 8, 8 + ph), _PANEL, -1)
    for i, txt in enumerate(lines):
        cv2.putText(
            frame, txt,
            (W - pw, 8 + 26 + i * 28),
            _FONT, 0.63, _CYAN, 1,
        )
