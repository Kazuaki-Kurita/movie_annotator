from __future__ import annotations

import json
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import cv2

from .models import VALID_CLASSES, Flower, Observation

YOLO_CLASS_TO_ID = {class_name: index for index, class_name in enumerate(VALID_CLASSES)}


@dataclass(frozen=True, slots=True)
class YoloExportResult:
    output_dir: Path
    image_count: int
    label_count: int
    annotation_count: int
    skipped_observation_count: int


def _normalise_bbox(
    observation: Observation,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float] | None:
    """Convert a pixel bbox to a clipped YOLO bbox.

    YOLO detection labels use normalised ``x_center y_center width height``.
    The input coordinates are stored in the original video resolution.
    """
    if image_width <= 0 or image_height <= 0:
        return None

    x_min = max(0, min(int(observation.x_min), image_width))
    y_min = max(0, min(int(observation.y_min), image_height))
    x_max = max(0, min(int(observation.x_max), image_width))
    y_max = max(0, min(int(observation.y_max), image_height))
    if x_max <= x_min or y_max <= y_min:
        return None

    x_center = ((x_min + x_max) / 2.0) / image_width
    y_center = ((y_min + y_max) / 2.0) / image_height
    width = (x_max - x_min) / image_width
    height = (y_max - y_min) / image_height
    return x_center, y_center, width, height


def observation_to_yolo_line(
    observation: Observation,
    flower: Flower,
    image_width: int,
    image_height: int,
) -> str | None:
    class_id = YOLO_CLASS_TO_ID.get(flower.class_name)
    normalised = _normalise_bbox(observation, image_width, image_height)
    if class_id is None or normalised is None:
        return None
    x_center, y_center, width, height = normalised
    return (
        f"{class_id} {x_center:.8f} {y_center:.8f} "
        f"{width:.8f} {height:.8f}"
    )


def export_yolo_dataset(
    output_root: Path,
    video_path: str | Path,
    flowers: Iterable[Flower],
    observations: Iterable[Observation],
) -> YoloExportResult:
    """Export raw video frames and paired YOLO label files.

    Output layout::

        output_root/
          yolo_dataset/
            images/<video>_frame_00000000.jpg
            labels/<video>_frame_00000000.txt
            classes.txt
            export_info.json

    All valid observations sharing the same frame are written to the same TXT.
    A temporary directory is used so an interrupted export does not leave a
    half-updated dataset.  The previous generated ``yolo_dataset`` directory is
    replaced only after the new export completes.
    """
    output_root = Path(output_root)
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"YOLO出力元の動画が見つかりません: {video_path}")

    flower_map: Mapping[str, Flower] = {
        flower.flower_id: flower for flower in flowers
    }
    grouped: dict[int, list[Observation]] = defaultdict(list)
    skipped = 0
    for observation in observations:
        if observation.frame_id is None or observation.frame_id < 0:
            skipped += 1
            continue
        flower = flower_map.get(observation.flower_id)
        if flower is None or flower.class_name not in YOLO_CLASS_TO_ID:
            skipped += 1
            continue
        grouped[observation.frame_id].append(observation)

    dataset_dir = output_root / "yolo_dataset"
    staging_dir = output_root / ".yolo_dataset_staging"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    images_dir = staging_dir / "images"
    labels_dir = staging_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise RuntimeError(f"YOLO画像抽出用に動画を開けません: {video_path}")

    image_count = 0
    annotation_count = 0
    video_stem = video_path.stem
    try:
        for frame_id in sorted(grouped):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ok, frame = capture.read()
            if not ok or frame is None:
                skipped += len(grouped[frame_id])
                continue

            image_height, image_width = frame.shape[:2]
            lines: list[str] = []
            for observation in sorted(
                grouped[frame_id],
                key=lambda item: (item.flower_id, item.pass_id),
            ):
                flower = flower_map[observation.flower_id]
                line = observation_to_yolo_line(
                    observation,
                    flower,
                    image_width,
                    image_height,
                )
                if line is None:
                    skipped += 1
                    continue
                lines.append(line)

            # This application exports only annotated frames.  If every bbox on
            # the frame is invalid, do not leave an unlabelled image behind.
            if not lines:
                continue

            basename = f"{video_stem}_frame_{frame_id:08d}"
            image_path = images_dir / f"{basename}.jpg"
            label_path = labels_dir / f"{basename}.txt"
            if not cv2.imwrite(
                str(image_path),
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, 95],
            ):
                raise RuntimeError(f"画像を書き込めません: {image_path}")
            label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            image_count += 1
            annotation_count += len(lines)
    finally:
        capture.release()

    (staging_dir / "classes.txt").write_text(
        "\n".join(VALID_CLASSES) + "\n",
        encoding="utf-8",
    )
    export_info = {
        "format": "YOLO detection",
        "source_video": str(video_path.resolve()),
        "image_extension": ".jpg",
        "jpeg_quality": 95,
        "class_names": {
            str(class_id): class_name
            for class_name, class_id in YOLO_CLASS_TO_ID.items()
        },
        "image_count": image_count,
        "label_count": image_count,
        "annotation_count": annotation_count,
        "skipped_observation_count": skipped,
        "note": "Images are raw video frames; bounding boxes are not drawn into them.",
    }
    (staging_dir / "export_info.json").write_text(
        json.dumps(export_info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    staging_dir.rename(dataset_dir)
    return YoloExportResult(
        output_dir=dataset_dir,
        image_count=image_count,
        label_count=image_count,
        annotation_count=annotation_count,
        skipped_observation_count=skipped,
    )
