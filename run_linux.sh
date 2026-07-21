#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi

source .venv/bin/activate

# PySide6でGUIを描画するため、Qt同梱版のOpenCVは使わない。
# opencv-pythonが既存環境に残っていると、cv2/qt/pluginsがPySide6と競合する。
python -m pip uninstall -y opencv-python opencv-contrib-python >/dev/null 2>&1 || true
python -m pip install -r requirements.txt

# 外部環境やOpenCVが設定したQtプラグインパスを引き継がない。
unset QT_PLUGIN_PATH || true
unset QT_QPA_PLATFORM_PLUGIN_PATH || true
unset QT_QPA_FONTDIR || true

python main.py "$@"
