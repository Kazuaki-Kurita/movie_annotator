from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from flower_gt_app.models import Flower, Observation, VideoMetadata, short_diameter_px
from flower_gt_app.storage import (
    build_section_rows,
    load_flowers,
    load_observations,
    next_flower_id,
    save_flowers,
    save_observations,
    save_sections,
)
from flower_gt_app.validator import validate_dataset


class StorageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.flowers = [
            Flower("GT_S00_001", 0, "bloom", "certain", 1, 1, 0, ""),
            Flower("GT_S00_002", 0, "faded", "certain", 1, 0, 0, "帰りは葉"),
            Flower("GT_S01_001", 1, "spent", "certain", 1, 1, 1, "境界"),
        ]
        self.observations = [
            Observation(
                "GT_S00_001",
                "outbound",
                3150,
                52552.5,
                1200,
                850,
                1305,
                970,
                "full",
                "",
            ),
            Observation(
                "GT_S00_001",
                "return",
                22150,
                369536.833,
                1188,
                855,
                1298,
                974,
                "full",
                "",
            ),
        ]

    def test_next_flower_id(self) -> None:
        self.assertEqual(next_flower_id(0, self.flowers), "GT_S00_003")
        self.assertEqual(next_flower_id(2, self.flowers), "GT_S02_001")

    def test_short_diameter_px(self) -> None:
        self.assertEqual(short_diameter_px((10, 20, 110, 70)), 50)
        self.assertEqual(short_diameter_px((10, 20, 40, 220)), 30)

    def test_section_rows_have_all_24_sections(self) -> None:
        rows = build_section_rows(self.flowers)
        self.assertEqual(len(rows), 24)
        self.assertEqual(rows[0]["bloom"], 1)
        self.assertEqual(rows[0]["faded"], 1)
        self.assertEqual(rows[0]["total"], 2)
        self.assertEqual(rows[23]["total"], 0)

    def test_csv_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            save_flowers(root / "ground_truth_flowers.csv", self.flowers)
            save_observations(root / "ground_truth_observations.csv", self.observations)
            save_sections(root / "ground_truth_sections.csv", self.flowers)

            loaded_flowers = load_flowers(root / "ground_truth_flowers.csv")
            loaded_observations = load_observations(root / "ground_truth_observations.csv")
            self.assertEqual(loaded_flowers, self.flowers)
            self.assertEqual(loaded_observations, self.observations)

            with (root / "ground_truth_sections.csv").open(
                encoding="utf-8", newline=""
            ) as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 24)
            self.assertEqual(rows[0]["left_marker_id"], "0")
            self.assertEqual(rows[0]["right_marker_id"], "1")


class ValidationTests(unittest.TestCase):
    def test_valid_dataset(self) -> None:
        video = VideoMetadata("video.mp4", 3840, 2160, 60000 / 1001, 30000)
        flowers = [Flower("GT_S00_001", 0, "bloom", "certain", 1, 0, 0, "")]
        observations = [
            Observation(
                "GT_S00_001",
                "outbound",
                3150,
                52552.5,
                100,
                200,
                300,
                400,
                "full",
                "",
            )
        ]
        issues = validate_dataset(flowers, observations, video)
        self.assertFalse([issue for issue in issues if issue.level == "error"])

    def test_duplicate_pass_is_error(self) -> None:
        video = VideoMetadata("video.mp4", 3840, 2160, 60.0, 1000)
        flowers = [Flower("GT_S00_001", 0, "bloom", "certain", 1, 0, 0, "")]
        base = Observation(
            "GT_S00_001", "outbound", 10, 166.667, 1, 1, 10, 10, "full", ""
        )
        duplicate = Observation(
            "GT_S00_001", "outbound", 20, 333.333, 1, 1, 10, 10, "full", ""
        )
        issues = validate_dataset(flowers, [base, duplicate], video)
        self.assertTrue(any("複数" in issue.message for issue in issues))

    def test_both_invisible_is_error(self) -> None:
        flower = Flower("GT_S00_001", 0, "bloom", "certain", 0, 0, 0, "")
        issues = validate_dataset([flower], [], None)
        self.assertTrue(any("少なくとも一方" in issue.message for issue in issues))


class YoloExportTests(unittest.TestCase):
    def test_observation_to_yolo_line(self) -> None:
        from flower_gt_app.yolo_export import observation_to_yolo_line

        flower = Flower("GT_S00_001", 0, "bloom", "certain", 1, 0, 0, "")
        observation = Observation(
            "GT_S00_001", "outbound", 1, 100.0, 10, 5, 30, 25, "full", ""
        )
        self.assertEqual(
            observation_to_yolo_line(observation, flower, 100, 50),
            "0 0.20000000 0.30000000 0.20000000 0.40000000",
        )

    def test_export_groups_annotations_by_frame(self) -> None:
        import cv2
        import numpy as np

        from flower_gt_app.yolo_export import export_yolo_dataset

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            video_path = root / "sample.avi"
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"MJPG"),
                5.0,
                (64, 48),
            )
            self.assertTrue(writer.isOpened())
            for value in (20, 80, 140):
                frame = np.full((48, 64, 3), value, dtype=np.uint8)
                writer.write(frame)
            writer.release()

            flowers = [
                Flower("GT_S00_001", 0, "bloom", "certain", 1, 0, 0, ""),
                Flower("GT_S00_002", 0, "spent", "certain", 1, 0, 0, ""),
                Flower("GT_S00_003", 0, "unripe", "certain", 1, 0, 0, ""),
            ]
            observations = [
                Observation(
                    "GT_S00_001", "outbound", 1, 200.0, 0, 0, 32, 24, "full", ""
                ),
                Observation(
                    "GT_S00_002", "outbound", 1, 200.0, 32, 24, 64, 48, "full", ""
                ),
                Observation(
                    "GT_S00_003", "outbound", 2, 400.0, 16, 12, 48, 36, "full", ""
                ),
            ]

            result = export_yolo_dataset(root, video_path, flowers, observations)
            self.assertEqual(result.image_count, 2)
            self.assertEqual(result.label_count, 2)
            self.assertEqual(result.annotation_count, 3)
            self.assertEqual(result.skipped_observation_count, 0)

            images = sorted((root / "yolo_dataset" / "images").glob("*.jpg"))
            labels = sorted((root / "yolo_dataset" / "labels").glob("*.txt"))
            self.assertEqual([path.stem for path in images], [path.stem for path in labels])
            first_lines = labels[0].read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(first_lines), 2)
            self.assertTrue(first_lines[0].startswith("0 "))
            self.assertTrue(first_lines[1].startswith("2 "))
            self.assertEqual(
                (root / "yolo_dataset" / "classes.txt").read_text(encoding="utf-8"),
                "bloom\nfaded\nspent\nunripe\n",
            )


if __name__ == "__main__":
    unittest.main()


def test_yolo_result_timestamp_maps_to_zero_based_frame(tmp_path):
    from flower_gt_app.yolo_results import load_yolo_results

    csv_path = tmp_path / "sample.csv"
    csv_path.write_text(
        "frame_id,timestamp,class_id,class_name,confidence,x_min,y_min,x_max,y_max\n"
        "65,00:00:01.067,1,faded,0.4124,10,20,30,40\n",
        encoding="utf-8",
    )

    results, skipped = load_yolo_results(
        csv_path,
        fps=60.0,
        frame_count=200,
        frame_width=100,
        frame_height=100,
    )

    assert skipped == 0
    assert list(results) == [64]
    assert results[64][0].bbox == (10, 20, 30, 40)
