from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import QSignalBlocker, Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSlider,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .models import (
    VALID_CLASSES,
    VALID_LABEL_QUALITIES,
    VALID_PASS_IDS,
    VALID_VISIBILITIES,
    Flower,
    Observation,
    VideoMetadata,
    short_diameter_px,
)
from .storage import (
    load_flowers,
    load_observations,
    next_flower_id,
    save_flowers,
    save_observations,
    save_sections,
)
from .validator import ValidationIssue, validate_dataset
from .video import VideoReader
from .yolo_export import YoloExportResult, export_yolo_dataset
from .yolo_results import YoloDetection, load_yolo_results
from .widgets import VideoCanvas


class MainWindow(QMainWindow):
    def __init__(
        self,
        video_path: str | None = None,
        output_dir: str | None = None,
    ) -> None:
        super().__init__()
        self.setWindowTitle("Flower Ground Truth Annotator")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.reader = VideoReader()
        self.video_metadata: VideoMetadata | None = None
        self.output_dir: Path | None = Path(output_dir).resolve() if output_dir else None
        self.flowers: dict[str, Flower] = {}
        self.observations: dict[tuple[str, str], Observation] = {}
        self.current_frame_id = -1
        self.current_frame = None
        self._slider_dragging = False
        self._loading_form = False
        self._loading_observation = False
        self.last_yolo_export_result: YoloExportResult | None = None
        self.yolo_results: dict[int, list[YoloDetection]] = {}
        self.yolo_csv_path: Path | None = None
        self.yolo_skipped_rows = 0
        self.yolo_overlay_enabled = False

        self.play_timer = QTimer(self)
        self.play_timer.timeout.connect(self._play_tick)

        self._build_ui()
        self._apply_initial_window_size()
        self._build_menu()
        self._build_shortcuts()
        self._update_output_label()

        if self.output_dir:
            self._load_project_data(self.output_dir)
        if video_path:
            self.open_video(Path(video_path))
        elif self.output_dir:
            self._try_open_video_from_project()

    def _apply_initial_window_size(self) -> None:
        """Choose an initial size that always fits inside the usable desktop area.

        The previous fixed 1600 x 950 size could exceed a laptop display's
        available geometry.  Qt then propagated the children's minimum size to
        the top-level window, which also interfered with normal maximisation.
        """
        screen = QApplication.primaryScreen()
        if screen is None:
            self.resize(1280, 800)
            return

        available = screen.availableGeometry()
        width = min(1600, max(900, int(available.width() * 0.92)))
        height = min(950, max(600, int(available.height() * 0.92)))
        width = min(width, available.width())
        height = min(height, available.height())
        self.resize(width, height)

    # ---------- UI construction ----------

    def _build_ui(self) -> None:
        central = QWidget(self)
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)

        top_bar = QHBoxLayout()
        self.video_label = QLabel("動画: 未選択")
        self.output_label = QLabel("出力: 未選択")
        # Long file-system paths must not become the minimum width of the
        # entire window.  Full paths remain available through tooltips.
        for path_label in (self.video_label, self.output_label):
            path_label.setMinimumWidth(0)
            path_label.setSizePolicy(
                QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
            )
        open_video_btn = QPushButton("動画を開く")
        open_video_btn.clicked.connect(self.choose_video)
        choose_output_btn = QPushButton("出力フォルダ")
        choose_output_btn.clicked.connect(self.choose_output_dir)
        save_btn = QPushButton("保存と検査")
        save_btn.clicked.connect(self.save_and_validate)
        top_bar.addWidget(open_video_btn)
        top_bar.addWidget(choose_output_btn)
        top_bar.addWidget(save_btn)
        top_bar.addWidget(self.video_label, 1)
        top_bar.addWidget(self.output_label, 1)
        root.addLayout(top_bar)

        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_splitter.addWidget(self._build_video_panel())
        self.main_splitter.addWidget(self._build_right_panel())
        self.main_splitter.setStretchFactor(0, 3)
        self.main_splitter.setStretchFactor(1, 2)
        self.main_splitter.setChildrenCollapsible(False)
        self.main_splitter.setSizes([960, 640])
        root.addWidget(self.main_splitter, 1)
        self.setCentralWidget(central)
        self.statusBar().showMessage("動画と出力フォルダを選択してください")

    def _build_video_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)

        self.canvas = VideoCanvas()
        self.canvas.bbox_changed.connect(self._bbox_changed)
        layout.addWidget(self.canvas, 1)

        self.position_slider = QSlider(Qt.Orientation.Horizontal)
        self.position_slider.setRange(0, 0)
        self.position_slider.sliderPressed.connect(self._slider_pressed)
        self.position_slider.sliderReleased.connect(self._slider_released)
        layout.addWidget(self.position_slider)

        controls = QHBoxLayout()
        for text, delta in (("-60", -60), ("-10", -10), ("-1", -1)):
            button = QPushButton(text)
            button.clicked.connect(lambda checked=False, d=delta: self.step_frames(d))
            controls.addWidget(button)

        self.play_button = QPushButton("再生")
        self.play_button.clicked.connect(self.toggle_playback)
        controls.addWidget(self.play_button)

        for text, delta in (("+1", 1), ("+10", 10), ("+60", 60)):
            button = QPushButton(text)
            button.clicked.connect(lambda checked=False, d=delta: self.step_frames(d))
            controls.addWidget(button)

        controls.addWidget(QLabel("再生時のフレーム間隔"))
        self.play_stride_combo = QComboBox()
        self.play_stride_combo.addItems(["1", "2", "4", "8"])
        self.play_stride_combo.setCurrentText("2")
        self.play_stride_combo.currentTextChanged.connect(self._update_timer_interval)
        controls.addWidget(self.play_stride_combo)
        controls.addStretch(1)

        self.yolo_display_button = QPushButton("YOLO検出を表示")
        self.yolo_display_button.setCheckable(True)
        self.yolo_display_button.setEnabled(False)
        self.yolo_display_button.toggled.connect(self._toggle_yolo_display)
        controls.addWidget(self.yolo_display_button)

        clear_bbox_btn = QPushButton("矩形を消去 [B]")
        clear_bbox_btn.clicked.connect(self.canvas.clear_bbox)
        controls.addWidget(clear_bbox_btn)
        layout.addLayout(controls)

        info = QHBoxLayout()
        self.frame_info_label = QLabel("frame: - / timestamp: -")
        self.video_info_label = QLabel("解像度・FPS: -")
        self.bbox_label = QLabel("bbox: 未指定")
        info.addWidget(self.frame_info_label)
        info.addWidget(self.video_info_label)
        info.addWidget(self.bbox_label, 1)
        layout.addLayout(info)
        return panel

    def _build_right_panel(self) -> QWidget:
        tabs = QTabWidget()
        tabs.setMinimumSize(0, 0)
        tabs.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        tabs.addTab(self._build_annotation_tab(), "登録")
        tabs.addTab(self._build_data_tab(), "データ・検査")
        return tabs

    def _build_annotation_tab(self) -> QWidget:
        tab = QWidget()
        tab.setMinimumSize(0, 0)
        tab_layout = QVBoxLayout(tab)
        tab_layout.setContentsMargins(4, 4, 4, 4)

        vertical_splitter = QSplitter(Qt.Orientation.Vertical)
        vertical_splitter.setChildrenCollapsible(False)

        # The input controls can scroll on a low-height display.  This prevents
        # their combined size hint from forcing the main window below the
        # desktop while preserving a large flower-list area.
        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setFrameShape(QFrame.Shape.NoFrame)
        controls_scroll.setMinimumHeight(120)
        controls_scroll.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        controls = QWidget()
        controls.setMinimumSize(0, 0)
        layout = QVBoxLayout(controls)
        layout.setContentsMargins(0, 0, 4, 0)

        form_group = QGroupBox("花情報")
        form = QFormLayout(form_group)

        self.pass_combo = QComboBox()
        self.pass_combo.addItems(VALID_PASS_IDS)
        self.pass_combo.currentTextChanged.connect(self._pass_changed)
        form.addRow("pass_id", self.pass_combo)

        self.section_spin = QSpinBox()
        self.section_spin.setRange(0, 23)
        self.section_spin.valueChanged.connect(self._section_changed)
        form.addRow("section_id", self.section_spin)

        self.proposed_id_label = QLabel("GT_S00_001")
        form.addRow("次の flower_id", self.proposed_id_label)

        self.class_combo = QComboBox()
        self.class_combo.addItems(VALID_CLASSES)
        form.addRow("class_name", self.class_combo)

        self.quality_combo = QComboBox()
        self.quality_combo.addItems(VALID_LABEL_QUALITIES)
        form.addRow("label_quality", self.quality_combo)

        visibility_row = QWidget()
        visibility_layout = QHBoxLayout(visibility_row)
        visibility_layout.setContentsMargins(0, 0, 0, 0)
        self.visible_outbound_check = QCheckBox("outbound")
        self.visible_return_check = QCheckBox("return")
        self.visible_outbound_check.setChecked(True)
        visibility_layout.addWidget(self.visible_outbound_check)
        visibility_layout.addWidget(self.visible_return_check)
        visibility_layout.addStretch(1)
        form.addRow("visible", visibility_row)

        self.boundary_check = QCheckBox("マーカー中心と花中心が一致")
        form.addRow("on_boundary", self.boundary_check)

        self.flower_notes_edit = QTextEdit()
        self.flower_notes_edit.setPlaceholderText("花単位の注記")
        self.flower_notes_edit.setMaximumHeight(75)
        form.addRow("flower notes", self.flower_notes_edit)
        layout.addWidget(form_group)

        obs_group = QGroupBox("新規登録用／選択花の代表観測")
        obs_form = QFormLayout(obs_group)
        self.obs_visibility_combo = QComboBox()
        self.obs_visibility_combo.addItems(VALID_VISIBILITIES)
        obs_form.addRow("visibility", self.obs_visibility_combo)
        self.obs_notes_edit = QTextEdit()
        self.obs_notes_edit.setPlaceholderText("葉で一部隠れている、など")
        self.obs_notes_edit.setMaximumHeight(75)
        obs_form.addRow("observation notes", self.obs_notes_edit)
        layout.addWidget(obs_group)

        new_button = QPushButton("新しい花＋現在の観測を登録 [N]")
        new_button.setMinimumHeight(36)
        new_button.clicked.connect(lambda: self.register_new_flower(with_observation=True))
        layout.addWidget(new_button)

        update_flower_button = QPushButton("選択中の花情報の更新")
        update_flower_button.setMinimumHeight(36)
        update_flower_button.clicked.connect(self.update_selected_flower)
        layout.addWidget(update_flower_button)

        delete_flower_button = QPushButton("選択中の花情報の削除")
        delete_flower_button.setMinimumHeight(36)
        delete_flower_button.clicked.connect(self.delete_selected_flower)
        layout.addWidget(delete_flower_button)
        layout.addStretch(1)
        controls_scroll.setWidget(controls)
        vertical_splitter.addWidget(controls_scroll)

        list_panel = QWidget()
        list_panel.setMinimumSize(0, 100)
        list_layout = QVBoxLayout(list_panel)
        list_layout.setContentsMargins(0, 0, 0, 0)
        self.quick_flower_table = self._make_flower_table()
        self.quick_flower_table.setMinimumSize(0, 80)
        self.quick_flower_table.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        list_layout.addWidget(QLabel("花一覧（選択すると登録フレームへ移動）"))
        list_layout.addWidget(self.quick_flower_table, 1)
        vertical_splitter.addWidget(list_panel)

        vertical_splitter.setStretchFactor(0, 0)
        vertical_splitter.setStretchFactor(1, 1)
        vertical_splitter.setSizes([500, 360])
        tab_layout.addWidget(vertical_splitter, 1)
        return tab

    def _build_data_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.flower_table = self._make_flower_table()
        layout.addWidget(QLabel("ground_truth_flowers.csv"))
        layout.addWidget(self.flower_table, 2)

        self.observation_table = QTableWidget(0, 7)
        self.observation_table.setHorizontalHeaderLabels(
            ["flower_id", "pass", "frame", "timestamp_ms", "bbox", "visibility", "notes"]
        )
        self.observation_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.observation_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.observation_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.observation_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.observation_table.horizontalHeader().setStretchLastSection(True)
        self.observation_table.itemDoubleClicked.connect(lambda _: self.load_selected_observation())
        layout.addWidget(QLabel("選択花の ground_truth_observations.csv"))
        layout.addWidget(self.observation_table, 1)

        obs_buttons = QHBoxLayout()
        load_obs_btn = QPushButton("観測位置へ移動")
        load_obs_btn.clicked.connect(self.load_selected_observation)
        obs_buttons.addWidget(load_obs_btn)
        obs_buttons.addStretch(1)
        layout.addLayout(obs_buttons)

        validate_btn = QPushButton("保存して全データを検査")
        validate_btn.clicked.connect(self.save_and_validate)
        layout.addWidget(validate_btn)

        self.validation_output = QPlainTextEdit()
        self.validation_output.setReadOnly(True)
        self.validation_output.setPlaceholderText("検査結果を表示します")
        self.validation_output.setMaximumHeight(220)
        layout.addWidget(self.validation_output)
        return tab

    def _make_flower_table(self) -> QTableWidget:
        table = QTableWidget(0, 8)
        table.setHorizontalHeaderLabels(
            ["flower_id", "section", "class", "quality", "out", "ret", "boundary", "notes"]
        )
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        table.horizontalHeader().setStretchLastSection(True)
        table.itemSelectionChanged.connect(lambda t=table: self._flower_selection_changed(t))
        return table

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("ファイル")
        open_video_action = QAction("動画を開く", self)
        open_video_action.setShortcut(QKeySequence.StandardKey.Open)
        open_video_action.triggered.connect(self.choose_video)
        file_menu.addAction(open_video_action)

        output_action = QAction("出力フォルダを選択", self)
        output_action.triggered.connect(self.choose_output_dir)
        file_menu.addAction(output_action)

        save_action = QAction("保存と検査", self)
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        save_action.triggered.connect(self.save_and_validate)
        file_menu.addAction(save_action)

        file_menu.addSeparator()
        exit_action = QAction("終了", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

    def _build_shortcuts(self) -> None:
        self._shortcuts: list[QShortcut] = []

        def add_shortcut(sequence: QKeySequence | str, callback) -> None:
            shortcut = QShortcut(QKeySequence(sequence), self)
            shortcut.activated.connect(callback)
            self._shortcuts.append(shortcut)

        add_shortcut(QKeySequence(Qt.Key.Key_Space), self.toggle_playback)
        add_shortcut(QKeySequence(Qt.Key.Key_Left), lambda: self.step_frames(-1))
        add_shortcut(QKeySequence(Qt.Key.Key_Right), lambda: self.step_frames(1))
        add_shortcut("Shift+Left", lambda: self.step_frames(-10))
        add_shortcut("Shift+Right", lambda: self.step_frames(10))
        add_shortcut("Ctrl+Left", lambda: self.step_frames(-60))
        add_shortcut("Ctrl+Right", lambda: self.step_frames(60))
        add_shortcut("B", self.canvas.clear_bbox)
        add_shortcut("N", lambda: self.register_new_flower(True))

    # ---------- Video ----------

    def choose_video(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "原動画を選択",
            str(Path.home()),
            "Video (*.mp4 *.MP4 *.mov *.MOV *.avi *.mkv);;All files (*)",
        )
        if filename:
            self.open_video(Path(filename))

    def open_video(self, path: Path) -> None:
        self.pause_playback()
        try:
            metadata = self.reader.open(path)
        except Exception as exc:
            QMessageBox.critical(self, "動画エラー", str(exc))
            return

        self.video_metadata = metadata
        self.video_label.setText(f"動画: {Path(metadata.path).name}")
        self.video_label.setToolTip(metadata.path)
        self.video_info_label.setText(
            f"{metadata.width}×{metadata.height} / {metadata.fps:.6f} fps / {metadata.frame_count:,} frames"
        )
        self.position_slider.setRange(0, max(0, metadata.frame_count - 1))
        self._update_timer_interval()
        self._load_yolo_results_for_video(path)
        self.seek_to_frame(0)

        if self.output_dir is None:
            self.output_dir = path.parent / f"{path.stem}_ground_truth"
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self._load_project_data(self.output_dir)
            self._update_output_label()

        self._save_project_metadata()
        resolution_note = ""
        if (metadata.width, metadata.height) != (3840, 2160):
            resolution_note = "（注意: 3840×2160ではありません）"

        if self.yolo_display_button.isEnabled():
            detection_count = sum(len(items) for items in self.yolo_results.values())
            yolo_note = f" / YOLO検出CSV: {detection_count:,}件"
        elif self.yolo_csv_path is not None and self.yolo_csv_path.exists():
            yolo_note = " / YOLO検出CSV: 読込失敗"
        else:
            yolo_note = " / YOLO検出CSV: なし"
        self.statusBar().showMessage(
            f"動画を開きました {resolution_note}{yolo_note}"
        )

    def seek_to_frame(self, frame_id: int, bbox: tuple[int, int, int, int] | None = None) -> None:
        if self.video_metadata is None:
            return
        self.pause_playback()
        result = self.reader.read_frame(frame_id)
        if result is None:
            self.statusBar().showMessage("フレームを読み込めませんでした")
            return
        actual_id, frame = result
        self._show_frame(actual_id, frame, clear_bbox=bbox is None)
        if bbox is not None:
            self.canvas.set_bbox(bbox, emit=True)

    def _show_frame(self, frame_id: int, frame, clear_bbox: bool = True) -> None:
        self.current_frame_id = frame_id
        self.current_frame = frame
        self.canvas.set_frame(frame)
        if clear_bbox:
            self.canvas.set_bbox(None, emit=True)
        with QSignalBlocker(self.position_slider):
            self.position_slider.setValue(frame_id)
        if self.video_metadata:
            timestamp = self.video_metadata.timestamp_ms_for_frame(frame_id)
            self.frame_info_label.setText(
                f"frame: {frame_id:,} / timestamp: {timestamp:.3f} ms"
            )
        self._update_yolo_overlay()

    def _load_yolo_results_for_video(self, video_path: Path) -> None:
        """Automatically load ``<video stem>.csv`` beside the video file."""
        self.yolo_results = {}
        self.yolo_csv_path = video_path.with_suffix(".csv")
        self.yolo_skipped_rows = 0
        self.yolo_overlay_enabled = False
        self.canvas.set_yolo_detections([])
        with QSignalBlocker(self.yolo_display_button):
            self.yolo_display_button.setChecked(False)
        self.yolo_display_button.setText("YOLO検出を表示")

        if self.video_metadata is None or not self.yolo_csv_path.exists():
            self.yolo_display_button.setEnabled(False)
            self.yolo_display_button.setToolTip(
                f"動画と同じ場所に {self.yolo_csv_path.name} がありません"
            )
            return

        try:
            results, skipped = load_yolo_results(
                self.yolo_csv_path,
                fps=self.video_metadata.fps,
                frame_count=self.video_metadata.frame_count,
                frame_width=self.video_metadata.width,
                frame_height=self.video_metadata.height,
            )
        except Exception as exc:
            self.yolo_display_button.setEnabled(False)
            self.yolo_display_button.setToolTip(str(exc))
            QMessageBox.warning(
                self,
                "YOLO検出CSV",
                f"{self.yolo_csv_path.name} を読み込めませんでした。\n{exc}",
            )
            return

        self.yolo_results = results
        self.yolo_skipped_rows = skipped
        detection_count = sum(len(items) for items in results.values())
        self.yolo_display_button.setEnabled(True)
        self.yolo_display_button.setToolTip(
            f"{self.yolo_csv_path.name}: {detection_count:,}件"
        )
        skipped_note = f" / 読み飛ばし {skipped:,}行" if skipped else ""
        self.statusBar().showMessage(
            f"YOLO検出CSVを読み込みました: {detection_count:,}件{skipped_note}"
        )

    def _toggle_yolo_display(self, checked: bool) -> None:
        self.yolo_overlay_enabled = checked
        self.yolo_display_button.setText(
            "YOLO検出を非表示" if checked else "YOLO検出を表示"
        )
        self._update_yolo_overlay(show_status=True)

    def _update_yolo_overlay(self, show_status: bool = False) -> None:
        detections: list[YoloDetection] = []
        if self.yolo_overlay_enabled and self.current_frame_id >= 0:
            detections = self.yolo_results.get(self.current_frame_id, [])
        self.canvas.set_yolo_detections(detections)

        if show_status and self.yolo_overlay_enabled:
            self.statusBar().showMessage(
                f"現在時刻のYOLO検出結果: {len(detections)}件"
            )

    def step_frames(self, delta: int) -> None:
        if self.video_metadata is None:
            return
        target = self.current_frame_id + delta
        target = max(0, min(target, self.video_metadata.frame_count - 1))
        self.seek_to_frame(target)

    def toggle_playback(self) -> None:
        if self.video_metadata is None:
            return
        if self.play_timer.isActive():
            self.pause_playback()
        else:
            self.canvas.set_bbox(None, emit=True)
            self._update_timer_interval()
            self.play_timer.start()
            self.play_button.setText("停止")

    def pause_playback(self) -> None:
        self.play_timer.stop()
        self.play_button.setText("再生")

    def _play_tick(self) -> None:
        stride = int(self.play_stride_combo.currentText())
        result = None
        for _ in range(stride):
            result = self.reader.read_next()
            if result is None:
                break
        if result is None:
            self.pause_playback()
            return
        frame_id, frame = result
        self._show_frame(frame_id, frame, clear_bbox=True)

    def _update_timer_interval(self) -> None:
        if self.video_metadata is None:
            return
        stride = int(self.play_stride_combo.currentText())
        interval = max(1, int(round(1000.0 * stride / self.video_metadata.fps)))
        self.play_timer.setInterval(interval)

    def _slider_pressed(self) -> None:
        self._slider_dragging = True
        self.pause_playback()

    def _slider_released(self) -> None:
        self._slider_dragging = False
        self.seek_to_frame(self.position_slider.value())

    # ---------- Project / storage ----------

    def choose_output_dir(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self,
            "CSV出力フォルダを選択",
            str(self.output_dir or Path.home()),
        )
        if not directory:
            return
        new_dir = Path(directory).resolve()
        if (new_dir / "ground_truth_flowers.csv").exists() and self.flowers:
            answer = QMessageBox.question(
                self,
                "既存データ",
                "選択したフォルダのCSVを読み込み、現在の画面データを置き換えますか？",
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        self.output_dir = new_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._load_project_data(self.output_dir)
        self._try_open_video_from_project()
        self._update_output_label()
        self.statusBar().showMessage("出力フォルダを設定しました")

    def _load_project_data(self, directory: Path) -> None:
        try:
            flowers = load_flowers(directory / "ground_truth_flowers.csv")
            observations = load_observations(directory / "ground_truth_observations.csv")
        except Exception as exc:
            QMessageBox.critical(self, "CSV読込エラー", str(exc))
            return
        flower_ids = [flower.flower_id for flower in flowers]
        if len(flower_ids) != len(set(flower_ids)):
            QMessageBox.critical(
                self, "CSV読込エラー", "ground_truth_flowers.csv に重複した flower_id があります"
            )
            return
        observation_keys = [
            (observation.flower_id, observation.pass_id) for observation in observations
        ]
        if len(observation_keys) != len(set(observation_keys)):
            QMessageBox.critical(
                self,
                "CSV読込エラー",
                "ground_truth_observations.csv に同じ花・同じ方向の重複があります",
            )
            return
        self.flowers = {flower.flower_id: flower for flower in flowers}
        self.observations = {
            (observation.flower_id, observation.pass_id): observation
            for observation in observations
        }
        self.refresh_tables()
        self._update_proposed_id()

    def _try_open_video_from_project(self) -> None:
        if self.output_dir is None or self.video_metadata is not None:
            return
        project_file = self.output_dir / "flower_gt_project.json"
        if not project_file.exists():
            return
        try:
            data = json.loads(project_file.read_text(encoding="utf-8"))
            video_path = Path(data.get("video_path", ""))
            if video_path.exists():
                self.open_video(video_path)
        except Exception:
            return

    def _ensure_output_dir(self) -> bool:
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            return True
        self.choose_output_dir()
        return self.output_dir is not None

    def _save_project_metadata(self) -> None:
        if self.output_dir is None:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        data: dict[str, object] = {"format_version": 1}
        if self.video_metadata:
            data.update(
                {
                    "video_path": self.video_metadata.path,
                    "width": self.video_metadata.width,
                    "height": self.video_metadata.height,
                    "fps": self.video_metadata.fps,
                    "frame_count": self.video_metadata.frame_count,
                }
            )
        (self.output_dir / "flower_gt_project.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def save_all(self, show_message: bool = False) -> bool:
        if not self._ensure_output_dir():
            return False
        assert self.output_dir is not None
        try:
            save_flowers(self.output_dir / "ground_truth_flowers.csv", self.flowers.values())
            save_observations(
                self.output_dir / "ground_truth_observations.csv",
                self.observations.values(),
            )
            save_sections(self.output_dir / "ground_truth_sections.csv", self.flowers.values())
            self._save_project_metadata()
        except Exception as exc:
            QMessageBox.critical(self, "保存エラー", str(exc))
            return False
        if show_message:
            self.statusBar().showMessage("3つのCSVをUTF-8で保存しました")
        return True

    def _autosave(self) -> None:
        if self.output_dir is not None:
            self.save_all(show_message=False)

    def save_and_validate(self) -> None:
        # Editing-time autosave remains CSV-only.  YOLO frame extraction can be
        # expensive for 4K video, so it runs only for the explicit
        # "保存と検査" action (including Ctrl+S).
        if not self.save_all(show_message=False):
            return
        issues = validate_dataset(
            self.flowers.values(), self.observations.values(), self.video_metadata
        )
        errors = [issue for issue in issues if issue.level == "error"]
        warnings = [issue for issue in issues if issue.level == "warning"]

        self.last_yolo_export_result = None
        yolo_error: str | None = None
        if errors:
            yolo_error = "検査エラーがあるため、YOLO形式は更新していません。"
        elif self.video_metadata is None:
            yolo_error = "動画が開かれていないため、YOLO形式を出力できません。"
        else:
            try:
                assert self.output_dir is not None
                self.last_yolo_export_result = export_yolo_dataset(
                    output_root=self.output_dir,
                    video_path=self.video_metadata.path,
                    flowers=self.flowers.values(),
                    observations=self.observations.values(),
                )
            except Exception as exc:
                yolo_error = str(exc)

        lines = [
            f"花: {len(self.flowers)} 件",
            f"観測: {len(self.observations)} 件",
            f"エラー: {len(errors)} 件 / 警告: {len(warnings)} 件",
        ]
        if self.last_yolo_export_result is not None:
            result = self.last_yolo_export_result
            lines.extend(
                [
                    f"YOLO画像: {result.image_count} 枚",
                    f"YOLO矩形: {result.annotation_count} 件",
                    f"YOLOスキップ: {result.skipped_observation_count} 件",
                    f"YOLO出力先: {result.output_dir}",
                ]
            )
        elif yolo_error:
            lines.append(f"YOLO出力: {yolo_error}")
        lines.append("")
        lines.extend(str(issue) for issue in issues)
        if not issues:
            lines.append("問題は見つかりませんでした。")
        self.validation_output.setPlainText("\n".join(lines))

        if errors:
            self.statusBar().showMessage("3つのCSVを保存しました。YOLO形式は未更新です。")
            QMessageBox.warning(
                self,
                "検査結果",
                f"CSVを保存しましたが、{len(errors)} 件のエラーがあります。"
                "YOLO形式は更新していません。データ・検査タブを確認してください。",
            )
        elif yolo_error:
            self.statusBar().showMessage("CSVは保存しましたが、YOLO形式の出力に失敗しました。")
            QMessageBox.warning(
                self,
                "YOLO保存結果",
                "3つのCSVは保存しましたが、YOLO形式を出力できませんでした。\n"
                f"{yolo_error}",
            )
        else:
            assert self.last_yolo_export_result is not None
            result = self.last_yolo_export_result
            skipped_note = (
                f"、スキップ{result.skipped_observation_count}件"
                if result.skipped_observation_count
                else ""
            )
            self.statusBar().showMessage(
                f"CSVとYOLO形式を保存しました（画像{result.image_count}枚、"
                f"矩形{result.annotation_count}件{skipped_note}）"
            )
            QMessageBox.information(
                self,
                "検査結果",
                f"保存完了。エラー0件、警告{len(warnings)}件です。\n"
                f"YOLO: 画像{result.image_count}枚 / 矩形{result.annotation_count}件"
                f"{skipped_note}",
            )

    def _update_output_label(self) -> None:
        if self.output_dir:
            self.output_label.setText(f"出力: {self.output_dir}")
            self.output_label.setToolTip(str(self.output_dir))
        else:
            self.output_label.setText("出力: 未選択")

    # ---------- Annotation ----------

    def _bbox_changed(self, bbox: object) -> None:
        if bbox is None:
            self.bbox_label.setText("bbox: 未指定")
        else:
            x_min, y_min, x_max, y_max = bbox  # type: ignore[misc]
            width = x_max - x_min
            height = y_max - y_min
            self.bbox_label.setText(
                f"bbox: ({x_min}, {y_min}) - ({x_max}, {y_max})  "
                f"{width}×{height} / 短径: {short_diameter_px((x_min, y_min, x_max, y_max))} px"
            )

    def _pass_changed(self, pass_id: str) -> None:
        if self._loading_form or self._loading_observation:
            return

        flower_id = self.selected_flower_id()
        if flower_id is not None:
            observation = self.observations.get((flower_id, pass_id))
            if observation is not None:
                self._display_observation(observation)
            else:
                self.canvas.clear_bbox()
                self.obs_notes_edit.clear()
                self.statusBar().showMessage(
                    f"{flower_id} には {pass_id} の代表観測が登録されていません"
                )
            return

        if pass_id == "outbound":
            self.visible_outbound_check.setChecked(True)
        else:
            self.visible_return_check.setChecked(True)

    def _section_changed(self, _: int) -> None:
        self._update_proposed_id()

    def _update_proposed_id(self) -> None:
        proposed = next_flower_id(self.section_spin.value(), self.flowers.values())
        self.proposed_id_label.setText(proposed)

    def _flower_from_form(self, flower_id: str) -> Flower:
        return Flower(
            flower_id=flower_id,
            section_id=self.section_spin.value(),
            class_name=self.class_combo.currentText(),
            label_quality=self.quality_combo.currentText(),
            visible_outbound=int(self.visible_outbound_check.isChecked()),
            visible_return=int(self.visible_return_check.isChecked()),
            on_boundary=int(self.boundary_check.isChecked()),
            notes=self.flower_notes_edit.toPlainText().strip(),
        )

    def _observation_from_current_frame(self, flower_id: str) -> Observation | None:
        if self.video_metadata is None or self.current_frame_id < 0:
            QMessageBox.warning(self, "観測登録", "原動画とフレームを選択してください")
            return None
        bbox = self.canvas.bbox
        if bbox is None:
            QMessageBox.warning(self, "観測登録", "花をドラッグして矩形を指定してください")
            return None
        x_min, y_min, x_max, y_max = bbox
        if not (
            0 <= x_min < x_max <= self.video_metadata.width
            and 0 <= y_min < y_max <= self.video_metadata.height
        ):
            QMessageBox.warning(self, "観測登録", "矩形座標が動画範囲外です")
            return None
        return Observation(
            flower_id=flower_id,
            pass_id=self.pass_combo.currentText(),
            frame_id=self.current_frame_id,
            timestamp_ms=self.video_metadata.timestamp_ms_for_frame(self.current_frame_id),
            x_min=x_min,
            y_min=y_min,
            x_max=x_max,
            y_max=y_max,
            visibility=self.obs_visibility_combo.currentText(),
            notes=self.obs_notes_edit.toPlainText().strip(),
        )

    def register_new_flower(self, with_observation: bool = True) -> None:
        if not self.visible_outbound_check.isChecked() and not self.visible_return_check.isChecked():
            QMessageBox.warning(
                self, "花登録", "行き・帰りの少なくとも一方を visible=1 にしてください"
            )
            return
        flower_id = next_flower_id(self.section_spin.value(), self.flowers.values())
        observation = None
        if with_observation:
            observation = self._observation_from_current_frame(flower_id)
            if observation is None:
                return

        flower = self._flower_from_form(flower_id)
        if observation:
            if observation.pass_id == "outbound":
                flower.visible_outbound = 1
            else:
                flower.visible_return = 1
        self.flowers[flower_id] = flower
        if observation:
            self.observations[(flower_id, observation.pass_id)] = observation
        self.refresh_tables(select_flower_id=flower_id)
        self._autosave()
        self._clear_notes_after_registration()
        self.canvas.clear_bbox()
        self._clear_flower_selection()
        self.statusBar().showMessage(f"{flower_id} を登録しました")

    def update_selected_flower(self) -> None:
        flower_id = self.selected_flower_id()
        if flower_id is None:
            QMessageBox.warning(self, "花情報更新", "花一覧から対象の花を選択してください")
            return
        if not self.visible_outbound_check.isChecked() and not self.visible_return_check.isChecked():
            QMessageBox.warning(
                self, "花情報更新", "行き・帰りの少なくとも一方を visible=1 にしてください"
            )
            return

        old = self.flowers[flower_id]
        if old.section_id != self.section_spin.value():
            QMessageBox.warning(
                self,
                "更新できない項目",
                "既登録花の section_id と flower_id は変更できません。\n"
                "区画・座標・観測時刻を変更する場合は、花情報を削除してから登録し直してください。",
            )
            self.section_spin.setValue(old.section_id)
            return

        updated = self._flower_from_form(flower_id)
        # Observation records are intentionally untouched. This operation updates
        # only flower-level attributes and never rewrites frame/time/bbox data.
        updated.section_id = old.section_id
        self.flowers[flower_id] = updated
        self.refresh_tables(select_flower_id=flower_id)
        self._autosave()
        self.statusBar().showMessage(
            f"{flower_id} の花情報を更新しました（観測時刻・座標は変更していません）"
        )

    def _clear_notes_after_registration(self) -> None:
        self.flower_notes_edit.clear()
        self.obs_notes_edit.clear()
        self.boundary_check.setChecked(False)
        self._update_proposed_id()

    # ---------- Tables / selection ----------

    def refresh_tables(self, select_flower_id: str | None = None) -> None:
        selected = select_flower_id or self.selected_flower_id()
        rows = sorted(self.flowers.values(), key=lambda flower: (flower.section_id, flower.flower_id))
        for table in (self.quick_flower_table, self.flower_table):
            table.blockSignals(True)
            table.setRowCount(len(rows))
            for row_index, flower in enumerate(rows):
                values = [
                    flower.flower_id,
                    str(flower.section_id),
                    flower.class_name,
                    flower.label_quality,
                    str(flower.visible_outbound),
                    str(flower.visible_return),
                    str(flower.on_boundary),
                    flower.notes,
                ]
                for column, value in enumerate(values):
                    table.setItem(row_index, column, QTableWidgetItem(value))
            table.blockSignals(False)
        if selected:
            self._select_flower_in_tables(selected, seek_observation=False)
        else:
            self.refresh_observation_table(None)
        self._update_proposed_id()

    def _select_flower_in_tables(
        self, flower_id: str, *, seek_observation: bool = False
    ) -> None:
        for table in (self.quick_flower_table, self.flower_table):
            for row in range(table.rowCount()):
                item = table.item(row, 0)
                if item and item.text() == flower_id:
                    table.blockSignals(True)
                    table.selectRow(row)
                    table.blockSignals(False)
                    break
        self._load_flower_into_form(flower_id)
        self.refresh_observation_table(flower_id)
        if seek_observation:
            self._load_preferred_observation_for_flower(flower_id)

    def _flower_selection_changed(self, source_table: QTableWidget) -> None:
        selected = source_table.selectedItems()
        if not selected:
            return
        flower_id = source_table.item(selected[0].row(), 0).text()
        self._select_flower_in_tables(flower_id, seek_observation=True)

    def _clear_flower_selection(self) -> None:
        for table in (self.quick_flower_table, self.flower_table):
            table.blockSignals(True)
            table.clearSelection()
            table.blockSignals(False)
        self.refresh_observation_table(None)

    def _preferred_observation(self, flower_id: str) -> Observation | None:
        preferred_pass = self.pass_combo.currentText()
        preferred = self.observations.get((flower_id, preferred_pass))
        if preferred is not None:
            return preferred
        return self.observations.get((flower_id, "outbound")) or self.observations.get(
            (flower_id, "return")
        )

    def _load_preferred_observation_for_flower(self, flower_id: str) -> None:
        observation = self._preferred_observation(flower_id)
        if observation is None:
            self.canvas.clear_bbox()
            self.obs_notes_edit.clear()
            self.statusBar().showMessage(f"{flower_id} には代表観測が登録されていません")
            return
        self._display_observation(observation)

    def _display_observation(self, observation: Observation) -> None:
        self._loading_observation = True
        try:
            with QSignalBlocker(self.pass_combo):
                self.pass_combo.setCurrentText(observation.pass_id)
            self.obs_visibility_combo.setCurrentText(observation.visibility)
            self.obs_notes_edit.setPlainText(observation.notes)
        finally:
            self._loading_observation = False

        bbox = (
            observation.x_min,
            observation.y_min,
            observation.x_max,
            observation.y_max,
        )
        target_frame: int | None = observation.frame_id
        if target_frame is None and observation.timestamp_ms is not None and self.video_metadata:
            target_frame = int(
                round(observation.timestamp_ms * self.video_metadata.fps / 1000.0)
            )

        if target_frame is not None and self.video_metadata is not None:
            self.seek_to_frame(target_frame, bbox=bbox)
            short_diameter = short_diameter_px(bbox)
            self.statusBar().showMessage(
                f"{observation.flower_id} / {observation.pass_id} の登録フレームへ移動 "
                f"（短径: {short_diameter} px）"
            )
        else:
            self.canvas.set_bbox(bbox, emit=True)
            self.statusBar().showMessage(
                f"{observation.flower_id} の観測はありますが、動画または時刻情報を利用できません"
            )

    def selected_flower_id(self) -> str | None:
        for table in (self.quick_flower_table, self.flower_table):
            selected = table.selectedItems()
            if selected:
                item = table.item(selected[0].row(), 0)
                if item and item.text() in self.flowers:
                    return item.text()
        return None

    def _load_flower_into_form(self, flower_id: str) -> None:
        flower = self.flowers.get(flower_id)
        if flower is None:
            return
        self._loading_form = True
        try:
            self.section_spin.setValue(flower.section_id)
            self.class_combo.setCurrentText(flower.class_name)
            self.quality_combo.setCurrentText(flower.label_quality)
            self.visible_outbound_check.setChecked(bool(flower.visible_outbound))
            self.visible_return_check.setChecked(bool(flower.visible_return))
            self.boundary_check.setChecked(bool(flower.on_boundary))
            self.flower_notes_edit.setPlainText(flower.notes)
        finally:
            self._loading_form = False

    def refresh_observation_table(self, flower_id: str | None) -> None:
        observations = [
            observation
            for observation in self.observations.values()
            if flower_id is None or observation.flower_id == flower_id
        ]
        observations.sort(key=lambda obs: (obs.flower_id, 0 if obs.pass_id == "outbound" else 1))
        self.observation_table.setRowCount(len(observations))
        for row, observation in enumerate(observations):
            bbox = f"{observation.x_min},{observation.y_min},{observation.x_max},{observation.y_max}"
            values = [
                observation.flower_id,
                observation.pass_id,
                "" if observation.frame_id is None else str(observation.frame_id),
                "" if observation.timestamp_ms is None else f"{observation.timestamp_ms:.3f}",
                bbox,
                observation.visibility,
                observation.notes,
            ]
            for column, value in enumerate(values):
                self.observation_table.setItem(row, column, QTableWidgetItem(value))

    def _selected_observation_key(self) -> tuple[str, str] | None:
        selected = self.observation_table.selectedItems()
        if not selected:
            return None
        row = selected[0].row()
        flower_item = self.observation_table.item(row, 0)
        pass_item = self.observation_table.item(row, 1)
        if flower_item is None or pass_item is None:
            return None
        return flower_item.text(), pass_item.text()

    def load_selected_observation(self) -> None:
        key = self._selected_observation_key()
        if key is None or key not in self.observations:
            QMessageBox.warning(self, "観測位置", "観測行を選択してください")
            return
        observation = self.observations[key]
        self._select_flower_in_tables(observation.flower_id, seek_observation=False)
        self._display_observation(observation)

    def delete_selected_flower(self) -> None:
        flower_id = self.selected_flower_id()
        if flower_id is None:
            QMessageBox.warning(self, "花情報削除", "花一覧から対象の花を選択してください")
            return
        observation_count = sum(1 for key in self.observations if key[0] == flower_id)
        answer = QMessageBox.question(
            self,
            "削除確認",
            f"{flower_id} を削除しますか？\n"
            f"関連する代表観測 {observation_count} 件（時刻・座標を含む）も同時に削除されます。",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        del self.flowers[flower_id]
        for key in list(self.observations):
            if key[0] == flower_id:
                del self.observations[key]
        self.canvas.clear_bbox()
        self.obs_notes_edit.clear()
        self.refresh_tables()
        self._autosave()
        self.statusBar().showMessage(f"{flower_id} を削除しました")

    # ---------- Lifecycle ----------

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self.pause_playback()
        if self.flowers and self.output_dir:
            self.save_all(show_message=False)
        self.reader.close()
        event.accept()
