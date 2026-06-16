"""
dataset.py
----------
Download the RPC dataset from Kaggle Hub, convert COCO annotations to
YOLO format, and sample a stratified subset.

Also supports any COCO-formatted custom dataset — just point
RPCDatasetConverter at your own annotation files and image directories.

Usage (RPC / Kaggle):
    converter = RPCDatasetConverter(output_dir="/content/rpc_subset")
    converter.download_rpc()
    converter.convert(subset_frac=0.10)
    yaml_path = converter.write_yaml()

Usage (custom COCO dataset):
    converter = RPCDatasetConverter(output_dir="/content/custom_subset")
    converter.convert(
        train_ann_path  = "/path/to/train_annotations.json",
        train_img_dir   = "/path/to/train/images",
        val_ann_path    = "/path/to/val_annotations.json",
        val_img_dir     = "/path/to/val/images",
        subset_frac     = 1.0,   # use all data
    )
    yaml_path = converter.write_yaml()
"""

from __future__ import annotations

import json
import math
import random
import shutil
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from .pricing import PriceCatalog


class RPCDatasetConverter:
    """
    End-to-end pipeline: download → convert → subset → write YAML.

    Parameters
    ----------
    output_dir : str | Path
        Root directory for the converted dataset.
    seed : int
        Random seed for reproducible subset sampling.
    """

    KAGGLE_DATASET = "diyer22/retail-product-checkout-dataset"

    def __init__(self, output_dir: str | Path = "/content/rpc_subset", seed: int = 42):
        self.output_dir = Path(output_dir)
        self.seed = seed
        self._raw_dir: Path | None = None
        self.categories: dict[int, str] = {}   # {coco_id: name}
        self.cat_remap:  dict[int, int] = {}   # {coco_id: yolo_idx}
        self.catalog:    PriceCatalog | None = None

    # ------------------------------------------------------------------
    # Step 1 — Download (RPC / Kaggle)
    # ------------------------------------------------------------------

    def download_rpc(self) -> Path:
        """
        Download the RPC dataset using kagglehub.
        Requires KAGGLE_USERNAME and KAGGLE_KEY env vars (or ~/.kaggle/kaggle.json).

        Returns
        -------
        Path to the downloaded dataset root.
        """
        import kagglehub
        path = kagglehub.dataset_download(self.KAGGLE_DATASET)
        self._raw_dir = Path(path)
        print(f"Dataset downloaded → {self._raw_dir}")
        return self._raw_dir

    def set_raw_dir(self, path: str | Path) -> None:
        """Use a pre-downloaded dataset directory instead of downloading."""
        self._raw_dir = Path(path)

    # ------------------------------------------------------------------
    # Step 2 — Convert + Subset
    # ------------------------------------------------------------------

    def convert(
        self,
        subset_frac: float = 0.10,
        train_ann_path: str | Path | None = None,
        train_img_dir:  str | Path | None = None,
        val_ann_path:   str | Path | None = None,
        val_img_dir:    str | Path | None = None,
    ) -> None:
        """
        Convert COCO annotations to YOLO format and sample a subset.

        If ann/img paths are not provided, they are auto-detected from
        self._raw_dir (set by download_rpc() or set_raw_dir()).

        Parameters
        ----------
        subset_frac    : fraction of images to keep per split (0 < x ≤ 1.0)
        train_ann_path : explicit path to train COCO JSON
        train_img_dir  : explicit path to train images directory
        val_ann_path   : explicit path to val COCO JSON
        val_img_dir    : explicit path to val images directory
        """
        random.seed(self.seed)

        # Resolve paths
        t_ann = Path(train_ann_path) if train_ann_path else self._find_ann(["train"])
        t_img = Path(train_img_dir)  if train_img_dir  else self._find_img(["train"])
        v_ann = Path(val_ann_path)   if val_ann_path   else self._find_ann(["val"])
        v_img = Path(val_img_dir)    if val_img_dir    else self._find_img(["val"])

        print(f"Train ann : {t_ann}")
        print(f"Train img : {t_img}")
        print(f"Val   ann : {v_ann}")
        print(f"Val   img : {v_img}")

        # Convert train — builds shared category/remap tables
        print(f"\n[1/2] Converting TRAIN (subset={subset_frac:.0%}) …")
        self.categories, self.cat_remap = self._convert_split(
            ann_path    = t_ann,
            img_src_dir = t_img,
            out_img_dir = self.output_dir / "images" / "train",
            out_lbl_dir = self.output_dir / "labels" / "train",
            subset_frac = subset_frac,
        )

        # Convert val — reuses same cat_remap for label consistency
        print(f"\n[2/2] Converting VAL (subset={subset_frac:.0%}) …")
        self._convert_split(
            ann_path    = v_ann,
            img_src_dir = v_img,
            out_img_dir = self.output_dir / "images" / "val",
            out_lbl_dir = self.output_dir / "labels" / "val",
            subset_frac = subset_frac,
            cat_remap   = self.cat_remap,
        )

        # Build price catalogue
        self.catalog = PriceCatalog.from_categories(self.categories, self.cat_remap)
        self.catalog.save(self.output_dir / "price_catalog.json")
        print(f"\nPrice catalogue saved → {self.output_dir / 'price_catalog.json'}")
        print(f"Total classes: {len(self.catalog)}")

    # ------------------------------------------------------------------
    # Step 3 — Write YAML
    # ------------------------------------------------------------------

    def write_yaml(self, yaml_name: str = "dataset.yaml") -> Path:
        """Write the Ultralytics dataset YAML and return its path."""
        if not self.catalog:
            raise RuntimeError("Run convert() before write_yaml().")

        class_names = self.catalog.class_names()
        yaml_content = (
            f"# YOLOv8x — Retail Product Pricing-Recognition\n"
            f"path: {self.output_dir}\n"
            f"train: images/train\n"
            f"val:   images/val\n\n"
            f"nc: {len(class_names)}\n"
            f"names: {class_names}\n"
        )
        yaml_path = self.output_dir / yaml_name
        yaml_path.write_text(yaml_content)
        print(f"Dataset YAML → {yaml_path}")
        return yaml_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_ann(self, keywords: list[str]) -> Path:
        """Recursively search raw_dir for annotation JSON matching keywords."""
        if self._raw_dir is None:
            raise RuntimeError("Call download_rpc() or set_raw_dir() first.")
        for kw in keywords:
            for p in self._raw_dir.rglob(f"*{kw}*.json"):
                return p
        raise FileNotFoundError(
            f"Could not auto-detect annotation file for keywords {keywords} "
            f"in {self._raw_dir}. Pass ann_path explicitly."
        )

    def _find_img(self, keywords: list[str]) -> Path:
        """Recursively search raw_dir for image directory matching keywords."""
        if self._raw_dir is None:
            raise RuntimeError("Call download_rpc() or set_raw_dir() first.")
        for kw in keywords:
            for p in self._raw_dir.rglob(f"*{kw}*"):
                if p.is_dir() and (
                    any(p.glob("*.jpg")) or any(p.glob("*.png"))
                ):
                    return p
        raise FileNotFoundError(
            f"Could not auto-detect image directory for keywords {keywords} "
            f"in {self._raw_dir}. Pass img_dir explicitly."
        )

    def _convert_split(
        self,
        ann_path:    Path,
        img_src_dir: Path,
        out_img_dir: Path,
        out_lbl_dir: Path,
        subset_frac: float = 1.0,
        cat_remap:   dict[int, int] | None = None,
    ) -> tuple[dict[int, str], dict[int, int]]:
        """
        Convert one COCO split to YOLO flat layout.

        Returns (categories, cat_remap) — only meaningful from the first
        call; subsequent calls should pass cat_remap to ensure consistency.
        """
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        with open(ann_path) as f:
            coco = json.load(f)

        categories = {c["id"]: c["name"] for c in coco["categories"]}

        if cat_remap is None:
            sorted_ids = sorted(categories.keys())
            cat_remap  = {cid: idx for idx, cid in enumerate(sorted_ids)}

        id2img   = {img["id"]: img for img in coco["images"]}
        img2anns: dict[int, list] = defaultdict(list)
        for ann in coco["annotations"]:
            img2anns[ann["image_id"]].append(ann)

        all_ids = list(id2img.keys())
        if subset_frac < 1.0:
            n       = max(1, math.ceil(len(all_ids) * subset_frac))
            all_ids = random.sample(all_ids, n)
        print(f"  {len(all_ids)} images selected")

        for img_id in tqdm(all_ids, desc="  Converting"):
            meta = id2img[img_id]
            src  = img_src_dir / meta["file_name"]
            if not src.exists():
                src = next(img_src_dir.rglob(meta["file_name"]), None)
                if src is None:
                    continue

            dst = out_img_dir / src.name
            if not dst.exists():
                shutil.copy2(src, dst)

            W, H = meta["width"], meta["height"]
            lines = []
            for ann in img2anns[img_id]:
                cid = ann["category_id"]
                if cid not in cat_remap:
                    continue
                x, y, w, h = ann["bbox"]
                cx = (x + w / 2) / W
                cy = (y + h / 2) / H
                lines.append(
                    f"{cat_remap[cid]} {cx:.6f} {cy:.6f} "
                    f"{w / W:.6f} {h / H:.6f}"
                )

            (out_lbl_dir / (src.stem + ".txt")).write_text("\n".join(lines))

        return categories, cat_remap
