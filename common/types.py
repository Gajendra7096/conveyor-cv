from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class BoxDetection:
    x: int
    y: int
    width: int
    height: int
    score: float = 1.0
    label: str | None = None
    mask: np.ndarray | None = None

    @property
    def x2(self) -> int:
        return self.x + self.width

    @property
    def y2(self) -> int:
        return self.y + self.height

    @property
    def center(self) -> tuple[float, float]:
        return self.x + self.width / 2.0, self.y + self.height / 2.0

    @property
    def area(self) -> int:
        return max(0, self.width) * max(0, self.height)

    def clipped(self, width: int, height: int) -> BoxDetection:
        x1 = max(0, min(self.x, width - 1))
        y1 = max(0, min(self.y, height - 1))
        x2 = max(x1 + 1, min(self.x2, width))
        y2 = max(y1 + 1, min(self.y2, height))
        return BoxDetection(
            x=x1,
            y=y1,
            width=x2 - x1,
            height=y2 - y1,
            score=self.score,
            label=self.label,
        )

    def to_dict(self) -> dict[str, Any]:
        center_x, center_y = self.center
        return {
            "x": int(self.x),
            "y": int(self.y),
            "width": int(self.width),
            "height": int(self.height),
            "center_x": float(center_x),
            "center_y": float(center_y),
            "score": float(self.score),
            "label": self.label,
        }
