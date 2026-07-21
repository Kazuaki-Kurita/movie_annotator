from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from .models import (
    VALID_CLASSES,
    VALID_LABEL_QUALITIES,
    VALID_PASS_IDS,
    VALID_VISIBILITIES,
    Flower,
    Observation,
    VideoMetadata,
)
from .storage import FLOWER_ID_PATTERN


@dataclass(slots=True)
class ValidationIssue:
    level: str  # "error" or "warning"
    source: str
    message: str

    def __str__(self) -> str:
        return f"[{self.level.upper()}] {self.source}: {self.message}"


def validate_dataset(
    flowers: Iterable[Flower],
    observations: Iterable[Observation],
    video: VideoMetadata | None,
) -> list[ValidationIssue]:
    flower_list = list(flowers)
    observation_list = list(observations)
    issues: list[ValidationIssue] = []

    flower_ids: set[str] = set()
    for flower in flower_list:
        source = flower.flower_id or "flower"
        if flower.flower_id in flower_ids:
            issues.append(ValidationIssue("error", source, "flower_id が重複しています"))
        flower_ids.add(flower.flower_id)

        match = FLOWER_ID_PATTERN.fullmatch(flower.flower_id)
        if not match:
            issues.append(
                ValidationIssue(
                    "warning", source, "推奨形式 GT_S区画_連番 と一致しません"
                )
            )
        elif int(match.group("section")) != flower.section_id:
            issues.append(
                ValidationIssue(
                    "error", source, "flower_id 内の区画番号と section_id が一致しません"
                )
            )

        if not 0 <= flower.section_id <= 23:
            issues.append(ValidationIssue("error", source, "section_id は 0〜23 です"))
        if flower.class_name not in VALID_CLASSES:
            issues.append(ValidationIssue("error", source, "class_name が不正です"))
        if flower.label_quality not in VALID_LABEL_QUALITIES:
            issues.append(ValidationIssue("error", source, "label_quality が不正です"))
        if flower.visible_outbound not in (0, 1) or flower.visible_return not in (0, 1):
            issues.append(
                ValidationIssue("error", source, "visible_outbound/return は 0 または 1 です")
            )
        if flower.visible_outbound == 0 and flower.visible_return == 0:
            issues.append(
                ValidationIssue(
                    "error", source, "行き・帰りの少なくとも一方を visible=1 にしてください"
                )
            )
        if flower.on_boundary not in (0, 1):
            issues.append(ValidationIssue("error", source, "on_boundary は 0 または 1 です"))

    observation_keys: set[tuple[str, str]] = set()
    observations_by_flower: dict[str, set[str]] = {}
    for observation in observation_list:
        source = f"{observation.flower_id}/{observation.pass_id}"
        if observation.flower_id not in flower_ids:
            issues.append(
                ValidationIssue("error", source, "ground_truth_flowers.csv に存在しない花IDです")
            )
        if observation.pass_id not in VALID_PASS_IDS:
            issues.append(ValidationIssue("error", source, "pass_id が不正です"))
        key = (observation.flower_id, observation.pass_id)
        if key in observation_keys:
            issues.append(
                ValidationIssue("error", source, "同じ花・同じ方向の観測が複数あります")
            )
        observation_keys.add(key)
        observations_by_flower.setdefault(observation.flower_id, set()).add(observation.pass_id)

        if observation.frame_id is None and observation.timestamp_ms is None:
            issues.append(
                ValidationIssue(
                    "error", source, "frame_id と timestamp_ms の少なくとも一方が必要です"
                )
            )
        if observation.frame_id is not None and observation.frame_id < 0:
            issues.append(ValidationIssue("error", source, "frame_id は 0 以上です"))
        if observation.timestamp_ms is not None and observation.timestamp_ms < 0:
            issues.append(ValidationIssue("error", source, "timestamp_ms は 0 以上です"))

        width = video.width if video else 3840
        height = video.height if video else 2160
        if not (
            0 <= observation.x_min < observation.x_max <= width
            and 0 <= observation.y_min < observation.y_max <= height
        ):
            issues.append(
                ValidationIssue(
                    "error",
                    source,
                    f"矩形座標が 0..{width}, 0..{height} の範囲外または大小関係が不正です",
                )
            )
        if observation.visibility not in VALID_VISIBILITIES:
            issues.append(ValidationIssue("error", source, "visibility が不正です"))

        if video:
            if observation.frame_id is not None and observation.frame_id >= video.frame_count:
                issues.append(
                    ValidationIssue("error", source, "frame_id が動画の総フレーム数以上です")
                )
            if observation.timestamp_ms is not None and observation.timestamp_ms > video.duration_ms:
                issues.append(
                    ValidationIssue("warning", source, "timestamp_ms が動画長を超えています")
                )
            if observation.frame_id is not None and observation.timestamp_ms is not None:
                expected = video.timestamp_ms_for_frame(observation.frame_id)
                difference = abs(expected - observation.timestamp_ms)
                frame_duration = 1000.0 / video.fps
                if difference > frame_duration:
                    issues.append(
                        ValidationIssue(
                            "error",
                            source,
                            f"frame_id と timestamp_ms が {difference:.3f} ms ずれています",
                        )
                    )
                elif difference > frame_duration * 0.5:
                    issues.append(
                        ValidationIssue(
                            "warning",
                            source,
                            f"frame_id と timestamp_ms が半フレーム以上ずれています ({difference:.3f} ms)",
                        )
                    )

    for flower in flower_list:
        passes = observations_by_flower.get(flower.flower_id, set())
        if flower.visible_outbound and "outbound" not in passes:
            issues.append(
                ValidationIssue(
                    "warning", flower.flower_id, "行きで visible=1 ですが代表観測がありません"
                )
            )
        if flower.visible_return and "return" not in passes:
            issues.append(
                ValidationIssue(
                    "warning", flower.flower_id, "帰りで visible=1 ですが代表観測がありません"
                )
            )
        if not passes:
            issues.append(
                ValidationIssue("warning", flower.flower_id, "代表観測が1件もありません")
            )

    if video:
        if (video.width, video.height) != (3840, 2160):
            issues.append(
                ValidationIssue(
                    "warning",
                    "video",
                    f"原動画解像度が 3840×2160 ではありません ({video.width}×{video.height})",
                )
            )
        if video.fps <= 0:
            issues.append(ValidationIssue("error", "video", "FPSを取得できません"))

    return issues
