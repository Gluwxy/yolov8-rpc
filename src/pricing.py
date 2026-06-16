"""
pricing.py
----------
Price catalogue for retail product categories.

Supports:
  - Loading / saving a JSON price catalogue
  - Assigning prices to COCO category names (keyword match + deterministic fallback)
  - Plug-in custom catalogue via JSON or a plain dict

Usage:
    catalog = PriceCatalog.from_json("data/price_catalog_default.json")
    price   = catalog.get_price(class_idx)
    catalog.assign_from_categories(categories_dict, cat_remap)
"""

from __future__ import annotations

import json
from pathlib import Path


# ---------------------------------------------------------------------------
# Keyword → price table (CNY).  Keys are substrings matched case-insensitively
# against the COCO category name.
# ---------------------------------------------------------------------------
_KEYWORD_PRICES: dict[str, float] = {
    # beverages
    "cola": 3.5, "pepsi": 3.5, "sprite": 3.5, "fanta": 3.5,
    "water": 2.0, "juice": 6.5, "milk": 5.5, "tea": 4.5,
    "coffee": 8.5, "beer": 7.0, "energy": 9.5, "yogurt": 6.0,
    "soda": 3.5, "drink": 4.0,
    # snacks
    "chips": 5.5, "crisp": 5.5, "cookies": 7.5, "candy": 4.0,
    "chocolate": 9.5, "popcorn": 4.5, "crackers": 6.5, "nuts": 12.0,
    "biscuit": 5.0, "wafer": 6.0, "gum": 2.5, "jelly": 4.5,
    # instant / packaged food
    "noodles": 4.5, "rice": 3.5, "porridge": 5.0, "soup": 8.0,
    "sauce": 10.0, "jam": 9.5, "honey": 18.0, "vinegar": 8.0,
    "oil": 25.0, "flour": 6.5,
    # personal care
    "shampoo": 18.0, "conditioner": 20.0, "soap": 9.5,
    "toothpaste": 12.0, "lotion": 22.0, "cream": 25.0,
    "facewash": 18.0, "deodorant": 15.0,
    # household
    "detergent": 15.0, "tissue": 10.0, "cleaner": 12.0,
    "freshener": 14.0, "bleach": 8.0,
}

# Deterministic price tiers cycled by category id when no keyword matches
_PRICE_TIERS: list[float] = [
    2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 12.0, 15.0,
]


class PriceCatalog:
    """
    Maps YOLO class index  →  {name, price}.

    Parameters
    ----------
    data : dict
        {str(class_idx): {"name": str, "price": float}}
    """

    def __init__(self, data: dict[str, dict] | None = None):
        self._data: dict[int, dict] = {}
        if data:
            for k, v in data.items():
                self._data[int(k)] = v

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_json(cls, path: str | Path) -> "PriceCatalog":
        """Load catalogue from a JSON file produced by assign_from_categories."""
        path = Path(path)
        with open(path) as f:
            raw = json.load(f)
        return cls(raw)

    @classmethod
    def from_categories(
        cls,
        categories: dict[int, str],   # {coco_cat_id: name}
        cat_remap: dict[int, int],     # {coco_cat_id: yolo_class_idx}
    ) -> "PriceCatalog":
        """
        Build catalogue automatically from COCO category metadata.

        categories : {coco_cat_id: category_name}
        cat_remap  : {coco_cat_id: yolo_class_idx}
        """
        inst = cls()
        for coco_id, yolo_idx in cat_remap.items():
            name  = categories.get(coco_id, f"class_{coco_id}")
            price = _assign_price(name, coco_id)
            inst._data[yolo_idx] = {"name": name, "price": price}
        return inst

    @classmethod
    def from_custom(cls, mapping: dict[str, float]) -> "PriceCatalog":
        """
        Build from a simple {class_name: price} dict.
        Class indices are assigned alphabetically.

        Example
        -------
        catalog = PriceCatalog.from_custom({"cola": 3.5, "chips": 5.5})
        """
        inst = cls()
        for idx, (name, price) in enumerate(sorted(mapping.items())):
            inst._data[idx] = {"name": name, "price": price}
        return inst

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    def get_price(self, class_idx: int) -> float:
        return self._data.get(class_idx, {}).get("price", 0.0)

    def get_name(self, class_idx: int) -> str:
        return self._data.get(class_idx, {}).get("name", f"class_{class_idx}")

    def class_names(self) -> list[str]:
        """Ordered list of class names (index 0, 1, 2 …)."""
        return [self._data[i]["name"] for i in sorted(self._data)]

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"PriceCatalog({len(self)} classes)"

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(
                {str(k): v for k, v in sorted(self._data.items())},
                f, indent=2, ensure_ascii=False,
            )

    # ------------------------------------------------------------------
    # Receipt helpers
    # ------------------------------------------------------------------

    def compute_receipt(
        self, class_counts: dict[int, int]
    ) -> tuple[list[tuple[str, int, float]], float]:
        """
        Given {class_idx: count}, return line items and grand total.

        Returns
        -------
        lines : [(name, count, subtotal), ...]
        total : float
        """
        lines = []
        total = 0.0
        for idx, count in sorted(class_counts.items()):
            name     = self.get_name(idx)
            price    = self.get_price(idx)
            subtotal = price * count
            total   += subtotal
            lines.append((name, count, subtotal))
        return lines, total


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _assign_price(name: str, cat_id: int) -> float:
    name_low = name.lower()
    for keyword, price in _KEYWORD_PRICES.items():
        if keyword in name_low:
            return price
    return _PRICE_TIERS[cat_id % len(_PRICE_TIERS)]
