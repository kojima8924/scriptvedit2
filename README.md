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
obj.time(6) <= move(x=0.5, y=0.5, anchor="center") \
               & scale(lambda u: lerp(0.5, 1, u)) \
               & fade(lambda u: u)
```

### Transformは静的、Effectは定数またはアニメーション

- **Transform** (`|` で連結、`<=` で適用): 1回だけ適用される空間変換
  - `resize(sx, sy)` ... サイズ変更

- **Effect** (`&` で連結、`<=` で適用): float で定数、lambda(u) でアニメーション
  - `move(x, y, anchor)` ... 配置位置（固定 or from/toアニメーション）
  - `scale(0.5)` ... 定数0.5倍
  - `scale(lambda u: lerp(0.5, 1, u))` ... 0.5倍 → 等倍にアニメーション
  - `fade(0.5)` ... 定数 半透明
  - `fade(lambda u: u)` ... 透明 → 不透明にアニメーション

`u` は正規化時間（0〜1）。Effectの表示開始から終了まで線形に変化する。

### Expr式ビルダー

lambda内で使える数学関数を多数提供。ffmpegのフィルタ式に自動コンパイルされる。

```python
# sin波フェード（フェードイン→フェードアウト）
fade(lambda u: sin(u * PI))

# 加速するスケール
scale(lambda u: lerp(0.5, 1, smoothstep(0, 1, u)))

# 円運動
move(x=lambda u: 0.5 + 0.3 * cos(u * 2 * PI),
     y=lambda u: 0.5 + 0.3 * sin(u * 2 * PI),
     anchor="center")
```

使用可能な関数:
- 三角: `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`
- 双曲線: `sinh`, `cosh`, `tanh`
- 指数/対数: `exp`, `log`, `sqrt`, `log10`, `cbrt`
- 丸め: `floor`, `ceil`, `trunc`
- 補間: `lerp(a, b, t)`
- クランプ: `clip(x, lo, hi)`, `clamp`
- ステップ: `step(edge, x)`, `smoothstep(edge0, edge1, x)`
- その他: `mod`, `frac`, `deg2rad`, `rad2deg`
- 組み込み互換: `abs`, `min`, `max`, `round`, `pow`
- 定数: `PI`, `E`

### move（位置・移動）

`move` は Effect として overlay の座標を制御する。

```python
# 固定位置
move(x=0.5, y=0.5, anchor="center")

# from/to 移動アニメーション
move(from_x=0.0, from_y=0.5, to_x=1.0, to_y=0.5, anchor="center")

# lambda 移動
move(x=lambda u: lerp(0.2, 0.8, u), y=0.5, anchor="center")
```

### キャッシュ

`Object.cache(path)` で単体レンダした結果をファイルに保存し、そのファイルを source に持つ新 Object を返す。

```python
# 画像キャッシュ（1フレーム）
cached = obj.cache("cached.png")

# 動画キャッシュ（duration分）
cached = obj.cache("cached.mp4")

# キャッシュ読み込み（既存ファイル）
obj = Object.load_cache("cached.mp4")
```

`overwrite=False` を指定すると、既存ファイルがあればレンダをスキップして高速化。

### anchor / pause / until（クロスレイヤー同期）

レイヤー間でタイミングを同期する仕組み。

```python
# レイヤーA: 幕を3秒表示してアンカーを打つ
maku = Object("maku.png")
maku.time(3) <= move(x=0.5, y=0.5, anchor="center")
anchor("curtain_done")

# レイヤーB: 幕が終わるまで待ってから登場
pause.until("curtain_done")
oni = Object("oni.png")
oni.time(3) <= move(x=0.5, y=0.5, anchor="center")
```

- `anchor(name)` ... 現在のタイムライン位置に名前付きマーカーを登録
- `pause.time(N)` ... N秒間の非描画待機
- `pause.until(name)` ... アンカー時刻まで非描画待機
- `obj.until(name)` ... durationをアンカー時刻まで伸長

### 2パスアーキテクチャ

`render()` は2段階で実行される:

1. **Plan pass** ... アンカーを固定点反復で解決（cache は no-op）
2. **Render pass** ... アンカー確定済みの状態で本実行、ffmpegコマンドを構築・実行

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
bg.time(6) <= move(x=0.5, y=0.5, anchor="center") \
              & scale(lambda u: lerp(1.5, 1, u)) \
              & fade(lambda u: u)
```

### 実行

```
python main.py
```

### dry_run（コマンド確認のみ）

```python
cmd = p.render("output.mp4", dry_run=True)
# ffmpegを実行せず、コマンドリスト（list[str]）を返す
```

### テスト

```
cd test
python test_snapshot.py        # スナップショットテスト（ffmpeg式の回帰検出）
python test_snapshot.py --update  # スナップショット更新
python test_errors.py          # エラーケーステスト
python test01_main.py          # 個別テスト（MP4生成）
```

## 依存

- Python 3.10+
- ffmpeg（PATHに必要）
