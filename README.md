# Flower Ground Truth Annotator

4K・60fps の原動画を人間が確認し、CSVとYOLO形式の正解データを作成するための Python デスクトップアプリです。

- `ground_truth_flowers.csv`
- `ground_truth_observations.csv`
- `ground_truth_sections.csv`

`ground_truth_sections.csv` は花単位データから常に自動生成されます。`保存と検査`を実行すると、追加で`yolo_dataset/images`と`yolo_dataset/labels`を生成します。

## 主な機能

- 原動画の再生、一時停止、1・10・60フレーム移動
- 動画上での矩形ドラッグ（赤色表示）
- 矩形四隅のハンドルをドラッグして拡大・縮小
- 単純なクリックでは矩形を消去せず、`矩形を消去 [B]` のみで削除
- 花一覧の選択時に、登録済みの代表観測フレームへ自動移動
- 登録矩形を動画上に表示
- 表示を縮小しても、矩形座標は原動画の解像度基準で保存
- `GT_S区画_連番` 形式の flower ID 自動採番
- 新規花と現在フレームの代表観測を同時登録
- 既登録花は花単位属性のみ更新し、flower ID・section・観測時刻・座標を固定
- flower IDを選択して、関連観測を含む花情報全体を削除
- 既存CSVの再読込と編集
- 編集のたびに3つのCSVを自動保存
- `保存と検査`または`Ctrl+S`でYOLO形式も追加出力
- 24区画すべてを含む区画別集計の自動生成
- 次の形式検査
  - flower ID の重複と区画番号の一致
  - section、class、quality、pass、visibility の値域
  - 行き・帰りの少なくとも一方が visible=1
  - 同じ花・同じ方向の観測重複
  - 観測から花への参照整合性
  - 矩形座標の範囲と大小関係
  - frame ID と timestamp の整合性
  - 動画の総フレーム数・解像度との整合性

## 推奨環境

- Python 3.11 または 3.12
- Windows 10/11 または Ubuntu 22.04/24.04
- 4K動画はローカルSSD上に配置することを推奨

## セットアップ

### Windows PowerShell

```powershell
cd flower_gt_annotator
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py
```

### Ubuntu / Linux

```bash
cd flower_gt_annotator
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py
```

UbuntuでQt関連の起動エラーが出る場合は、次のパッケージを追加してください。

```bash
sudo apt update
sudo apt install -y libegl1 libgl1 libxkbcommon-x11-0
```

## コマンドラインから動画と出力先を指定

```bash
python main.py \
  --video /path/to/DJI_20251029125546_0011_D.MP4 \
  --output-dir /path/to/work_directory
```

GUIから選択する場合、動画を先に開くと、動画と同じ場所に次の出力フォルダを自動作成します。

```text
DJI_20251029125546_0011_D_ground_truth/
```

## 基本操作

1. `動画を開く` から必ず原動画を選択します。
2. outbound または return を選びます。
3. section、class、label quality などを指定します。
4. 動画を停止し、花を囲むように左ドラッグします。登録後は赤い矩形の四隅をドラッグして大きさを調整できます。単純なクリックでは矩形は消えません。
5. 初めて登録する花は `新しい花＋現在の観測を登録` を押します。
6. 下側の花一覧から既存の flower ID を選ぶと、その花の登録フレームへ移動し、矩形が表示されます。
7. class、label quality、visible、boundary、notesを修正する場合は `選択中の花情報の更新` を押します。
8. flower ID、section、観測時刻、座標を変更する場合は `選択中の花情報の削除` を実行し、改めて新規登録します。
9. 最後に `保存と検査` を実行します。CSVに加え、YOLO形式の画像とTXTも出力されます。

登録タブで行えるデータ変更は、次の3種類です。

1. `新しい花＋現在の観測を登録`
2. `選択中の花情報の更新`
3. `選択中の花情報の削除`

花情報の更新では、`ground_truth_observations.csv` のframe ID、timestamp、bbox、visibility、notesは変更されません。

## キーボード操作

| キー | 動作 |
|---|---|
| Space | 再生・停止 |
| Left / Right | 1フレーム移動 |
| Shift + Left / Right | 10フレーム移動 |
| Ctrl + Left / Right | 60フレーム移動 |
| B | 矩形消去 |
| N | 新しい花＋現在観測を登録 |
| Ctrl + S | 保存と検査 |

## 境界規則

アプリでは section ID を人間が選択します。境界上の花は、次の規則に従って若い番号側へ割り当て、`on_boundary=1` にしてください。

- section 0: Marker 0 以上、Marker 1 以下
- section 1: Marker 1 より大きく、Marker 2 以下
- …
- section 23: Marker 23 より大きく、Marker 24 以下

したがって Marker 1 上は section 0、Marker 2 上は section 1 です。


## YOLO形式の出力

`保存と検査`または`Ctrl+S`を実行し、検査エラーが0件の場合に、出力フォルダ内の`yolo_dataset`を更新します。登録・更新時の自動保存は従来どおりCSVのみです。4K動画から全登録フレームを抽出する処理を毎回繰り返さないための仕様です。

- 画像とラベルは同じベース名で保存されます。
- 同じフレームに複数の花がある場合、1枚の画像に対応する1つのTXTへ複数行を保存します。
- 画像は原動画から抽出したJPEG（品質95）で、赤い矩形は画像へ焼き込みません。
- TXTの1行は `class_id x_center y_center width height` で、座標は0～1に正規化されます。
- クラスIDは `0=bloom`, `1=faded`, `2=spent`, `3=unripe` です。
- `classes.txt`と出力情報を記録した`export_info.json`も生成されます。
- YOLO出力は毎回作り直されるため、削除済みの花や古い座標が残りません。

ファイル名の例：

```text
images/DJI_20251029125546_0011_D_frame_00003150.jpg
labels/DJI_20251029125546_0011_D_frame_00003150.txt
```

## CSV文字コード

出力CSVは BOMなし UTF-8、カンマ区切り、ヘッダーありです。読込時は BOMありUTF-8にも対応します。

## 4K・60fps利用時の注意

- 表示はウィンドウに合わせて縮小されますが、保存座標は原動画基準です。
- 再生が重い場合は、`再生時のフレーム間隔` を 2、4、8 にします。
- アノテーション時は停止して1フレーム移動を使用してください。
- 動画シークは OpenCV の動画バックエンドを使用します。長距離シーク後は画面の frame ID 表示を確認してください。
- 可変フレームレート動画では `frame_id / fps` と実PTSが一致しない可能性があります。本用途ではDJI原動画が固定フレームレートであることを確認して使用してください。

## 出力例

```text
work_directory/
├── flower_gt_project.json
├── ground_truth_flowers.csv
├── ground_truth_observations.csv
├── ground_truth_sections.csv
└── yolo_dataset/
    ├── classes.txt
    ├── export_info.json
    ├── images/
    │   └── <video>_frame_00000000.jpg
    └── labels/
        └── <video>_frame_00000000.txt
```

`flower_gt_project.json` はアプリ再開用であり、評価用CSVには含まれません。

## テスト

```bash
python -m unittest discover -s tests -v
```

## Linuxで `Could not load the Qt platform plugin "xcb"` が出る場合

UbuntuではQtのX11プラグインが利用するシステムライブラリを追加してください。

```bash
sudo apt update
sudo apt install -y libxcb-cursor0
```

本アプリはPySide6で画面を表示するため、OpenCVはGUI機能を含まない
`opencv-python-headless`を使用します。古い環境に`opencv-python`が残っていても、
更新版の`run_linux.sh`が削除してから依存関係を入れ直します。

手動で修復する場合は次を実行します。

```bash
source .venv/bin/activate
python -m pip uninstall -y opencv-python opencv-contrib-python
python -m pip install --upgrade --force-reinstall opencv-python-headless
./run_linux.sh
```

## YOLO検出結果の参照表示

動画を開くと、動画と同じフォルダにある同名CSVを自動的に読み込みます。

- 動画: `2025_1031_top.mp4` または `2025_1031_top.MP4`
- 検出結果: `2025_1031_top.csv`

動画操作欄の「YOLO検出を表示」ボタンを押すと、現在時刻に対応する検出結果が緑色の破線矩形で表示されます。矩形にはクラス名と信頼度が表示されます。もう一度押すと非表示になります。

YOLO検出矩形は参照表示専用です。赤色の正解データ矩形とは独立しており、登録・編集・削除の対象にはなりません。CSVの時刻は `timestamp` 列を基準に動画FPSへ換算するため、CSVの `frame_id` が1始まりでもアプリの0始まりフレームへ対応します。

必要なCSV列は次のとおりです。

`timestamp, class_id, class_name, confidence, x_min, y_min, x_max, y_max`
