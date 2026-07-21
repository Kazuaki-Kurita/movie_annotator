from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QImage, QMouseEvent, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QSizePolicy, QWidget

from .yolo_results import YoloDetection


@dataclass(slots=True)
class DisplayTransform:
    target: QRectF
    source_width: int
    source_height: int

    def widget_to_source(self, point: QPoint) -> tuple[int, int] | None:
        if not self.target.contains(point):
            return None
        return self.widget_to_source_clamped(point)

    def widget_to_source_clamped(self, point: QPoint) -> tuple[int, int]:
        """Convert a widget point to a source point, clamping to the video area."""
        widget_x = min(max(float(point.x()), self.target.left()), self.target.right())
        widget_y = min(max(float(point.y()), self.target.top()), self.target.bottom())
        x_ratio = (widget_x - self.target.left()) / self.target.width()
        y_ratio = (widget_y - self.target.top()) / self.target.height()
        x = int(round(x_ratio * self.source_width))
        y = int(round(y_ratio * self.source_height))
        x = max(0, min(x, self.source_width))
        y = max(0, min(y, self.source_height))
        return x, y

    def source_rect_to_widget(self, bbox: tuple[int, int, int, int]) -> QRectF:
        x_min, y_min, x_max, y_max = bbox
        sx = self.target.width() / self.source_width
        sy = self.target.height() / self.source_height
        return QRectF(
            self.target.left() + x_min * sx,
            self.target.top() + y_min * sy,
            max(1.0, (x_max - x_min) * sx),
            max(1.0, (y_max - y_min) * sy),
        )


class VideoCanvas(QWidget):
    bbox_changed = Signal(object)

    _MIN_BBOX_SIZE = 2
    _HANDLE_HALF_SIZE = 3.0
    _HANDLE_HIT_MARGIN = 7.0

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        # Keep the 16:9 canvas usable while allowing the main window to fit
        # smaller displays and maximise without being constrained by children.
        self.setMinimumSize(320, 180)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._pixmap: QPixmap | None = None
        self._source_width = 0
        self._source_height = 0
        self._bbox: tuple[int, int, int, int] | None = None
        self._yolo_detections: list[YoloDetection] = []
        self._drag_start: tuple[int, int] | None = None
        self._drag_current: tuple[int, int] | None = None
        self._drag_mode: str | None = None  # "draw" or "resize"
        self._resize_handle: str | None = None
        self._bbox_before_drag: tuple[int, int, int, int] | None = None
        self._press_widget_position: QPointF | None = None
        self._drag_has_moved = False
        self._transform: DisplayTransform | None = None
        self._message = "動画を開いてください"

    @property
    def bbox(self) -> tuple[int, int, int, int] | None:
        return self._bbox

    def set_message(self, message: str) -> None:
        self._message = message
        self.update()

    def set_frame(self, frame_bgr: np.ndarray) -> None:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        bytes_per_line = channels * width
        image = QImage(
            rgb.data,
            width,
            height,
            bytes_per_line,
            QImage.Format.Format_RGB888,
        ).copy()
        self._pixmap = QPixmap.fromImage(image)
        self._source_width = width
        self._source_height = height
        self.update()

    def set_bbox(self, bbox: tuple[int, int, int, int] | None, emit: bool = False) -> None:
        self._bbox = bbox
        self._reset_drag_state()
        self.update()
        if emit:
            self.bbox_changed.emit(self._bbox)

    def clear_bbox(self) -> None:
        self.set_bbox(None, emit=True)

    def set_yolo_detections(self, detections: list[YoloDetection]) -> None:
        """Display model detections without changing the editable GT bbox."""
        self._yolo_detections = list(detections)
        self.update()

    def _reset_drag_state(self) -> None:
        self._drag_start = None
        self._drag_current = None
        self._drag_mode = None
        self._resize_handle = None
        self._bbox_before_drag = None
        self._press_widget_position = None
        self._drag_has_moved = False
        self.setCursor(Qt.CursorShape.ArrowCursor)

    def _compute_transform(self) -> DisplayTransform | None:
        if self._pixmap is None or self._source_width <= 0 or self._source_height <= 0:
            return None
        area = self.rect()
        scale = min(area.width() / self._source_width, area.height() / self._source_height)
        width = self._source_width * scale
        height = self._source_height * scale
        left = (area.width() - width) / 2.0
        top = (area.height() - height) / 2.0
        return DisplayTransform(QRectF(left, top, width, height), self._source_width, self._source_height)

    @staticmethod
    def _normalized_bbox(
        first: tuple[int, int], second: tuple[int, int]
    ) -> tuple[int, int, int, int]:
        x1, y1 = first
        x2, y2 = second
        return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)

    @classmethod
    def _is_valid_bbox(cls, bbox: tuple[int, int, int, int]) -> bool:
        return (
            bbox[2] - bbox[0] >= cls._MIN_BBOX_SIZE
            and bbox[3] - bbox[1] >= cls._MIN_BBOX_SIZE
        )

    def _drag_bbox(self) -> tuple[int, int, int, int] | None:
        if self._drag_start is None or self._drag_current is None:
            return None
        return self._normalized_bbox(self._drag_start, self._drag_current)

    def _corner_points(self, bbox: tuple[int, int, int, int]) -> dict[str, QPointF]:
        if self._transform is None:
            return {}
        rect = self._transform.source_rect_to_widget(bbox)
        return {
            "top_left": rect.topLeft(),
            "top_right": rect.topRight(),
            "bottom_right": rect.bottomRight(),
            "bottom_left": rect.bottomLeft(),
        }

    def _hit_test_handle(self, point: QPointF) -> str | None:
        if self._bbox is None:
            return None
        hit_radius = self._HANDLE_HALF_SIZE + self._HANDLE_HIT_MARGIN
        for name, corner in self._corner_points(self._bbox).items():
            if abs(point.x() - corner.x()) <= hit_radius and abs(point.y() - corner.y()) <= hit_radius:
                return name
        return None

    @staticmethod
    def _opposite_corner(
        bbox: tuple[int, int, int, int], handle: str
    ) -> tuple[int, int]:
        x_min, y_min, x_max, y_max = bbox
        opposites = {
            "top_left": (x_max, y_max),
            "top_right": (x_min, y_max),
            "bottom_right": (x_min, y_min),
            "bottom_left": (x_max, y_min),
        }
        return opposites[handle]

    @staticmethod
    def _cursor_for_handle(handle: str | None) -> Qt.CursorShape:
        if handle in {"top_left", "bottom_right"}:
            return Qt.CursorShape.SizeFDiagCursor
        if handle in {"top_right", "bottom_left"}:
            return Qt.CursorShape.SizeBDiagCursor
        return Qt.CursorShape.CrossCursor

    def paintEvent(self, event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.GlobalColor.black)
        self._transform = self._compute_transform()
        if self._pixmap is None or self._transform is None:
            painter.setPen(Qt.GlobalColor.white)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._message)
            return

        painter.drawPixmap(self._transform.target, self._pixmap, QRectF(self._pixmap.rect()))

        # YOLO detections are a non-editable reference overlay.  They use a
        # dashed green line so that the editable red ground-truth rectangle
        # remains visually distinct and keeps all existing mouse behaviour.
        yolo_color = QColor(0, 255, 80)
        for detection in self._yolo_detections:
            widget_rect = self._transform.source_rect_to_widget(detection.bbox)
            painter.save()
            painter.setBrush(QBrush(Qt.BrushStyle.NoBrush))
            painter.setPen(
                QPen(yolo_color, 2, Qt.PenStyle.DashLine)
            )
            painter.drawRect(widget_rect)

            label = f"{detection.class_name} {detection.confidence:.2f}"
            metrics = painter.fontMetrics()
            text_rect = QRectF(metrics.boundingRect(label))
            text_rect.adjust(-3.0, -1.0, 3.0, 1.0)
            label_x = widget_rect.left()
            label_y = max(self._transform.target.top(), widget_rect.top() - text_rect.height())
            text_rect.moveTopLeft(QPointF(label_x, label_y))
            painter.setPen(QPen(QColor(0, 0, 0, 0), 0))
            background = QColor(0, 0, 0, 190)
            painter.setBrush(QBrush(background))
            painter.drawRect(text_rect)
            painter.setPen(QPen(yolo_color, 1))
            painter.drawText(
                text_rect.adjusted(3.0, 1.0, -3.0, -1.0),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                label,
            )
            painter.restore()

        # A simple click must not make the existing rectangle disappear. A new
        # preview replaces it only after the pointer has moved far enough to
        # form a valid rectangle. Resize previews are shown immediately.
        bbox = self._bbox
        drag_bbox = self._drag_bbox()
        if drag_bbox is not None and (
            self._drag_mode == "resize" or self._is_valid_bbox(drag_bbox)
        ):
            bbox = drag_bbox

        if bbox and self._is_valid_bbox(bbox):
            widget_rect = self._transform.source_rect_to_widget(bbox)
            rectangle_color = QColor(255, 0, 0)
            painter.setPen(QPen(rectangle_color, 2))
            painter.drawRect(widget_rect)

            # Four small corner handles indicate that the rectangle can be
            # resized. Their hit area remains generous for easy operation.
            painter.setPen(QPen(Qt.GlobalColor.white, 1))
            painter.setBrush(QBrush(rectangle_color))
            for corner in self._corner_points(bbox).values():
                painter.drawRect(
                    QRectF(
                        corner.x() - self._HANDLE_HALF_SIZE,
                        corner.y() - self._HANDLE_HALF_SIZE,
                        self._HANDLE_HALF_SIZE * 2.0,
                        self._HANDLE_HALF_SIZE * 2.0,
                    )
                )

    def mousePressEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if event.button() != Qt.MouseButton.LeftButton or self._transform is None:
            return
        source = self._transform.widget_to_source(event.position().toPoint())
        if source is None:
            return

        self._bbox_before_drag = self._bbox
        self._press_widget_position = event.position()
        self._drag_has_moved = False
        handle = self._hit_test_handle(event.position())
        if handle is not None and self._bbox is not None:
            self._drag_mode = "resize"
            self._resize_handle = handle
            self._drag_start = self._opposite_corner(self._bbox, handle)
            self._drag_current = source
            self.setCursor(self._cursor_for_handle(handle))
        else:
            self._drag_mode = "draw"
            self._resize_handle = None
            self._drag_start = source
            self._drag_current = source
            self.setCursor(Qt.CursorShape.CrossCursor)
        self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if self._transform is None:
            return
        if self._drag_start is not None:
            if self._press_widget_position is not None:
                delta = event.position() - self._press_widget_position
                if abs(delta.x()) >= 2.0 or abs(delta.y()) >= 2.0:
                    self._drag_has_moved = True
            self._drag_current = self._transform.widget_to_source_clamped(
                event.position().toPoint()
            )
            self.update()
            return

        handle = self._hit_test_handle(event.position())
        self.setCursor(self._cursor_for_handle(handle))

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if event.button() != Qt.MouseButton.LeftButton or self._drag_start is None:
            return
        if self._press_widget_position is not None:
            delta = event.position() - self._press_widget_position
            if abs(delta.x()) >= 2.0 or abs(delta.y()) >= 2.0:
                self._drag_has_moved = True
        if self._transform is not None:
            self._drag_current = self._transform.widget_to_source_clamped(
                event.position().toPoint()
            )

        candidate = self._drag_bbox()
        previous = self._bbox_before_drag
        if (
            self._drag_has_moved
            and candidate is not None
            and self._is_valid_bbox(candidate)
        ):
            self._bbox = candidate
        else:
            # A click without a meaningful drag keeps the existing rectangle.
            # Deletion is intentionally available only through the B action.
            self._bbox = previous

        self._reset_drag_state()
        self.bbox_changed.emit(self._bbox)
        self.update()
