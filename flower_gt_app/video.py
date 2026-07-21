from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .models import VideoMetadata


class VideoReader:
    """Sequential playback and frame seeking through OpenCV.

    Frame IDs are zero-based. After a successful read, current_frame_id points to
    the frame that was returned.
    """

    def __init__(self) -> None:
        self.capture: cv2.VideoCapture | None = None
        self.metadata: VideoMetadata | None = None
        self.current_frame_id = -1

    def open(self, path: str | Path) -> VideoMetadata:
        self.close()
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise RuntimeError(f"動画を開けません: {path}")

        width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        if width <= 0 or height <= 0 or fps <= 0 or frame_count <= 0:
            capture.release()
            raise RuntimeError("動画メタデータを正しく取得できません")

        self.capture = capture
        self.metadata = VideoMetadata(
            path=str(Path(path).resolve()),
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
        )
        self.current_frame_id = -1
        return self.metadata

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()
        self.capture = None
        self.metadata = None
        self.current_frame_id = -1

    def read_next(self) -> tuple[int, np.ndarray] | None:
        if self.capture is None:
            return None
        ok, frame = self.capture.read()
        if not ok:
            return None
        reported_next = int(round(self.capture.get(cv2.CAP_PROP_POS_FRAMES)))
        self.current_frame_id = max(0, reported_next - 1)
        return self.current_frame_id, frame

    def read_frame(self, frame_id: int) -> tuple[int, np.ndarray] | None:
        if self.capture is None or self.metadata is None:
            return None
        frame_id = max(0, min(frame_id, self.metadata.frame_count - 1))
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, frame = self.capture.read()
        if not ok:
            return None
        reported_next = int(round(self.capture.get(cv2.CAP_PROP_POS_FRAMES)))
        actual_id = max(0, reported_next - 1)
        self.current_frame_id = actual_id
        return actual_id, frame
