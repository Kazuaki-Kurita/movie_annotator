from __future__ import annotations

import argparse
import os
import sys

from PySide6.QtCore import QLibraryInfo
from PySide6.QtWidgets import QApplication


def _configure_qt_plugin_path() -> None:
    """Prevent OpenCV's bundled Qt plugins from overriding PySide6 plugins."""
    for key in ("QT_PLUGIN_PATH", "QT_QPA_PLATFORM_PLUGIN_PATH", "QT_QPA_FONTDIR"):
        value = os.environ.get(key, "")
        if "cv2" in value.replace("\\", "/").lower():
            os.environ.pop(key, None)

    # Explicitly select the plugin directory shipped with the active PySide6.
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = QLibraryInfo.path(
        QLibraryInfo.LibraryPath.PluginsPath
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="4K flower ground-truth annotation GUI")
    parser.add_argument("--video", help="Path to the original video")
    parser.add_argument("--output-dir", help="Directory for the three CSV files")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _configure_qt_plugin_path()

    # Import after the Qt plugin path is fixed. MainWindow imports OpenCV internally.
    from flower_gt_app.main_window import MainWindow

    # OpenCV may alter Qt-related environment variables when a GUI-enabled wheel is
    # accidentally installed. Restore the PySide6 path immediately before QApplication.
    _configure_qt_plugin_path()

    app = QApplication(sys.argv)
    window = MainWindow(video_path=args.video, output_dir=args.output_dir)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
