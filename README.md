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

### 演算子によるDSL

- `|` (パイプ) ... Transform同士を連結して TransformChain を生成
- `&` (アンド) ... Effect同士を連結して EffectChain を生成
- `<=` (適用) ... Object に TransformChain / EffectChain を適用

```python
obj <= resize(sx=0.3, sy=0.3)
obj.time(6) <= move(x=0.5, y=0.5, anchor="center") & scale(1.5) & fade(alpha=0)
```

### Transformは静的、Effectはアニメーション

- **Transform** (`|` で連結、`<=` で適用): 1回だけ適用される空間変換
  - `resize(sx, sy)` ... サイズ変更

- **Effect** (`&` で連結、`<=` で適用): float引数が表示時間内で **指定値 → 1.0** に線形変化
  - `move(x, y, anchor)` ... 配置位置（固定 or from/toアニメーション）
  - `scale(1.5)` ... 1.5倍 → 等倍にアニメーション
  - `fade(alpha=0)` ... 透明 → 不透明にアニメーション

### move（位置・移動）

`move` は Effect として overlay の座標を制御する。

```python
# 固定位置
move(x=0.5, y=0.5, anchor="center")

# 移動アニメーション
move(from_x=0.0, from_y=0.5, to_x=1.0, to_y=0.5, anchor="center")
```

### キャッシュ

`Object.cache(path)` で単体レンダした結果をファイルに保存し、そのファイルを source に持つ新 Object を返す。

```python
# 画像キャッシュ（1フレーム）
img = (obj <= resize(sx=0.5, sy=0.5)).cache("cached.png")

# 動画キャッシュ（duration分）
vid = (obj.time(6) <= move(...) & scale(...) & fade(...)).cache("cached.mp4")

# キャッシュ読み込み（既存ファイル）
obj = Object.load_cache("cached.mp4")
```

`overwrite=False` を指定すると、既存ファイルがあればレンダをスキップして高速化。

### プリセット

TransformChain / EffectChain を変数に保存して再利用可能。

```python
preset_e = move(x=0.3, y=0.7, anchor="center") & scale(0.5) & fade(alpha=0)

obj = Object("image.png")
obj <= resize(sx=0.3, sy=0.3)
obj.time(6) <= preset_e
```

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
bg <= resize(sx=1, sy=1)
bg.time(6) <= move(x=0.5, y=0.5, anchor="center") & scale(1.5) & fade(alpha=0)
```

### 実行

```
python main.py
```

## 依存

- Python 3.10+
- ffmpeg（PATHに必要）
