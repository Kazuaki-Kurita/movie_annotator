from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

from .models import Flower, Observation

FLOWER_COLUMNS = [
    "flower_id",
    "section_id",
    "class_name",
    "label_quality",
    "visible_outbound",
    "visible_return",
    "on_boundary",
    "notes",
]

OBSERVATION_COLUMNS = [
    "flower_id",
    "pass_id",
    "frame_id",
    "timestamp_ms",
    "x_min",
    "y_min",
    "x_max",
    "y_max",
    "visibility",
    "notes",
]

SECTION_COLUMNS = [
    "section_id",
    "left_marker_id",
    "right_marker_id",
    "bloom",
    "faded",
    "spent",
    "unripe",
    "total",
]

FLOWER_ID_PATTERN = re.compile(r"^GT_S(?P<section>\d{2})_(?P<serial>\d{3,})$")


def _write_csv(path: Path, columns: list[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def save_flowers(path: Path, flowers: Iterable[Flower]) -> None:
    ordered = sorted(flowers, key=lambda flower: (flower.section_id, flower.flower_id))
    _write_csv(path, FLOWER_COLUMNS, (flower.to_dict() for flower in ordered))


def save_observations(path: Path, observations: Iterable[Observation]) -> None:
    def row(observation: Observation) -> dict[str, object]:
        data = observation.to_dict()
        data["frame_id"] = "" if observation.frame_id is None else observation.frame_id
        data["timestamp_ms"] = (
            "" if observation.timestamp_ms is None else f"{observation.timestamp_ms:.3f}"
        )
        return data

    ordered = sorted(
        observations,
        key=lambda observation: (
            observation.flower_id,
            0 if observation.pass_id == "outbound" else 1,
        ),
    )
    _write_csv(path, OBSERVATION_COLUMNS, (row(observation) for observation in ordered))


def build_section_rows(flowers: Iterable[Flower]) -> list[dict[str, int]]:
    counters: dict[int, Counter[str]] = {section_id: Counter() for section_id in range(24)}
    for flower in flowers:
        if 0 <= flower.section_id <= 23:
            counters[flower.section_id][flower.class_name] += 1

    rows: list[dict[str, int]] = []
    for section_id in range(24):
        counter = counters[section_id]
        bloom = counter["bloom"]
        faded = counter["faded"]
        spent = counter["spent"]
        unripe = counter["unripe"]
        rows.append(
            {
                "section_id": section_id,
                "left_marker_id": section_id,
                "right_marker_id": section_id + 1,
                "bloom": bloom,
                "faded": faded,
                "spent": spent,
                "unripe": unripe,
                "total": bloom + faded + spent + unripe,
            }
        )
    return rows


def save_sections(path: Path, flowers: Iterable[Flower]) -> None:
    _write_csv(path, SECTION_COLUMNS, build_section_rows(flowers))


def load_flowers(path: Path) -> list[Flower]:
    if not path.exists():
        return []
    flowers: list[Flower] = []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            flowers.append(
                Flower(
                    flower_id=row["flower_id"].strip(),
                    section_id=int(row["section_id"]),
                    class_name=row["class_name"].strip(),
                    label_quality=row["label_quality"].strip(),
                    visible_outbound=int(row["visible_outbound"]),
                    visible_return=int(row["visible_return"]),
                    on_boundary=int(row["on_boundary"]),
                    notes=row.get("notes", ""),
                )
            )
    return flowers


def _optional_int(value: str | None) -> int | None:
    value = (value or "").strip()
    return None if value == "" else int(value)


def _optional_float(value: str | None) -> float | None:
    value = (value or "").strip()
    return None if value == "" else float(value)


def load_observations(path: Path) -> list[Observation]:
    if not path.exists():
        return []
    observations: list[Observation] = []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            observations.append(
                Observation(
                    flower_id=row["flower_id"].strip(),
                    pass_id=row["pass_id"].strip(),
                    frame_id=_optional_int(row.get("frame_id")),
                    timestamp_ms=_optional_float(row.get("timestamp_ms")),
                    x_min=int(row["x_min"]),
                    y_min=int(row["y_min"]),
                    x_max=int(row["x_max"]),
                    y_max=int(row["y_max"]),
                    visibility=row["visibility"].strip(),
                    notes=row.get("notes", ""),
                )
            )
    return observations


def next_flower_id(section_id: int, flowers: Iterable[Flower]) -> str:
    max_serial = 0
    for flower in flowers:
        match = FLOWER_ID_PATTERN.fullmatch(flower.flower_id)
        if not match:
            continue
        if int(match.group("section")) == section_id:
            max_serial = max(max_serial, int(match.group("serial")))
    return f"GT_S{section_id:02d}_{max_serial + 1:03d}"
