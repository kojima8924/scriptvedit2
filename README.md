# scriptvedit2

Pythonスクリプトで動画を構成するDSL。ffmpegによるレンダリング。

## 設計思想

### 1ファイル = 1レイヤー

動画の各レイヤーを独立したPythonファイルとして管理する。
`main.py` は構成定義のみを担い、各レイヤーファイルの読み込み順序・重ね順を宣言する。

```
main.py        ... 構成定義（設定・レイヤー順序・レンダリング）
bg.py          ... 背景レイヤー
onigiri.py     ... 素材レイヤー
```

### 演算子によるチェーン記法

- `|` (パイプ) ... Transform（静的な空間変換: リサイズ、配置）
- `&` (アンド) ... Effect（時間アニメーション: スケール、フェード）

```python
obj | resize(sx=0.3, sy=0.3) | pos(x=0.5, y=0.5, anchor="center")
obj.time(6) & scale(1.5) & fade(alpha=0)
```

### Transformは静的、Effectはアニメーション

- **Transform** (`|`): 1回だけ適用される空間変換
  - `resize(sx, sy)` ... サイズ変更
  - `pos(x, y, anchor)` ... 配置位置

- **Effect** (`&`): float引数が表示時間内で **指定値 → 1.0** に線形変化
  - `scale(1.5)` ... 1.5倍 → 等倍にアニメーション
  - `fade(alpha=0)` ... 透明 → 不透明にアニメーション

### レイヤーの独立タイムライン

各レイヤーは0秒から独立したタイムラインを持つ。
動画全体のdurationは全レイヤーの最大値から自動算出される。

### priority による z-order 制御

`p.layer(filename, priority=N)` の `priority` で重ね順を制御する。
値が大きいほど手前に表示。記述順に依存しない。

## 使い方

### main.py（構成定義）

```python
from scriptvedit import *

p = Project()
p.configure(width=1280, height=720, fps=30, background_color="black")

p.layer("bg.py", priority=0)
p.layer("onigiri.py", priority=1)

p.render("output.mp4")
```

### レイヤーファイル（例: bg.py）

```python
from scriptvedit import *

bg = Object("bg_pattern_ishigaki.jpg")
bg | resize(sx=1, sy=1) | pos(x=0.5, y=0.5, anchor="center")
bg.time(6) & scale(1.5) & fade(alpha=0)
```

### 実行

```
python main.py
```

## 依存

- Python 3.10+
- ffmpeg（PATHに必要）
