from __future__ import annotations

import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class YoloDetection:
    """One YOLO detection associated with a zero-based video frame."""

    frame_id: int
    timestamp_ms: float
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]


_REQUIRED_COLUMNS = {
    "timestamp",
    "class_id",
    "class_name",
    "confidence",
    "x_min",
    "y_min",
    "x_max",
    "y_max",
}


def parse_timestamp_ms(value: str) -> float:
    """Parse HH:MM:SS.sss (or seconds as a number) into milliseconds."""

    text = str(value).strip()
    if not text:
        raise ValueError("timestamp が空です")

    parts = text.split(":")
    if len(parts) == 1:
        return float(parts[0]) * 1000.0
    if len(parts) != 3:
        raise ValueError(f"timestamp 形式が不正です: {text}")

    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = float(parts[2])
    if hours < 0 or not 0 <= minutes < 60 or not 0 <= seconds < 60:
        raise ValueError(f"timestamp の値が範囲外です: {text}")
    return ((hours * 60 + minutes) * 60 + seconds) * 1000.0


def load_yolo_results(
    csv_path: str | Path,
    fps: float,
    frame_count: int,
    frame_width: int,
    frame_height: int,
) -> tuple[dict[int, list[YoloDetection]], int]:
    """Load detections and index them by the app's zero-based frame ID.

    The model CSV's ``frame_id`` may be one-based.  Therefore ``timestamp`` is
    used as the source of truth and converted to a zero-based frame using the
    video's actual FPS.  Coordinates are clipped to the video boundaries.

    Returns ``(detections_by_frame, skipped_row_count)``.
    """

    if fps <= 0:
        raise ValueError("動画FPSが不正です")

    path = Path(csv_path)
    detections: dict[int, list[YoloDetection]] = defaultdict(list)
    skipped = 0

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("YOLO検出CSVにヘッダーがありません")

        missing = sorted(_REQUIRED_COLUMNS - set(reader.fieldnames))
        if missing:
            raise ValueError(
                "YOLO検出CSVに必要な列がありません: " + ", ".join(missing)
            )

        for row in reader:
            try:
                timestamp_ms = parse_timestamp_ms(row["timestamp"])
                frame_id = math.floor(timestamp_ms * fps / 1000.0 + 0.5)
                if frame_id < 0 or frame_id >= frame_count:
                    skipped += 1
                    continue

                x_min = max(0, min(int(float(row["x_min"])), frame_width))
                y_min = max(0, min(int(float(row["y_min"])), frame_height))
                x_max = max(0, min(int(float(row["x_max"])), frame_width))
                y_max = max(0, min(int(float(row["y_max"])), frame_height))
                if x_max <= x_min or y_max <= y_min:
                    skipped += 1
                    continue

                detection = YoloDetection(
                    frame_id=frame_id,
                    timestamp_ms=timestamp_ms,
                    class_id=int(float(row["class_id"])),
                    class_name=str(row["class_name"]).strip(),
                    confidence=float(row["confidence"]),
                    bbox=(x_min, y_min, x_max, y_max),
                )
            except (TypeError, ValueError, KeyError):
                skipped += 1
                continue

            detections[frame_id].append(detection)

    for frame_detections in detections.values():
        frame_detections.sort(key=lambda item: item.confidence, reverse=True)

    return dict(detections), skipped
