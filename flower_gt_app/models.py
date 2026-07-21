from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

VALID_CLASSES = ("bloom", "faded", "spent", "unripe")
VALID_LABEL_QUALITIES = ("certain", "uncertain")
VALID_PASS_IDS = ("outbound", "return")
VALID_VISIBILITIES = ("full", "partial")

BBox = tuple[int, int, int, int]


def short_diameter_px(bbox: BBox) -> int:
    """Return the shorter side length of an axis-aligned bounding box."""
    x_min, y_min, x_max, y_max = bbox
    return min(x_max - x_min, y_max - y_min)


@dataclass(slots=True)
class Flower:
    flower_id: str
    section_id: int
    class_name: str
    label_quality: str
    visible_outbound: int
    visible_return: int
    on_boundary: int
    notes: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class Observation:
    flower_id: str
    pass_id: str
    frame_id: Optional[int]
    timestamp_ms: Optional[float]
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    visibility: str
    notes: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class VideoMetadata:
    path: str
    width: int
    height: int
    fps: float
    frame_count: int

    @property
    def duration_ms(self) -> float:
        if self.fps <= 0:
            return 0.0
        return self.frame_count * 1000.0 / self.fps

    def timestamp_ms_for_frame(self, frame_id: int) -> float:
        if self.fps <= 0:
            raise ValueError("FPS must be positive")
        return frame_id * 1000.0 / self.fps
