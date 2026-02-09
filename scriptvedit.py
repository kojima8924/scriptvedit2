import subprocess


class Project:
    _current = None

    def __init__(self):
        self.width = 1920
        self.height = 1080
        self.fps = 30
        self.duration = None
        self.background_color = "black"
        self.objects = []
        self._layers = []  # [(start_idx, end_idx, priority)]
        Project._current = self

    def configure(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def layer(self, filename, priority=0):
        """レイヤーファイルを読み込み、Objectにpriorityを付与"""
        start_idx = len(self.objects)
        with open(filename, encoding="utf-8") as f:
            code = f.read()
        namespace = {}
        exec(compile(code, filename, "exec"), namespace)
        end_idx = len(self.objects)
        self._layers.append((start_idx, end_idx, priority))
        for obj in self.objects[start_idx:end_idx]:
            obj.priority = priority

    def _calc_total_duration(self):
        """各レイヤーの最大durationを返す"""
        max_dur = 0
        for start_idx, end_idx, _ in self._layers:
            layer_dur = 0
            for obj in self.objects[start_idx:end_idx]:
                if obj.duration is not None:
                    layer_dur += obj.duration
            max_dur = max(max_dur, layer_dur)
        return max_dur if max_dur > 0 else 5

    def render(self, output_path):
        if self.duration is None:
            self.duration = self._calc_total_duration()
        cmd = self._build_ffmpeg_cmd(output_path)
        print(f"実行コマンド:")
        print(f"  ffmpeg {' '.join(cmd[1:])}")
        print()
        subprocess.run(cmd, check=True)
        print(f"\n完了: {output_path}")

    def _build_ffmpeg_cmd(self, output_path):
        inputs = []
        filter_parts = []

        # [0] 背景色
        inputs.extend([
            "-f", "lavfi",
            "-i", f"color=c={self.background_color}:s={self.width}x{self.height}:d={self.duration}:r={self.fps}",
        ])

        # タイミング計算（レイヤーごとに独立、各レイヤーは0秒から開始）
        for start_idx, end_idx, _ in self._layers:
            current_time = 0
            for obj in self.objects[start_idx:end_idx]:
                obj.start_time = current_time
                if obj.duration is not None:
                    current_time += obj.duration

        # z-order: priorityでソート（小さい=奥、大きい=手前）
        sorted_objects = sorted(self.objects, key=lambda o: o.priority)

        current_base = "[0:v]"

        for i, obj in enumerate(sorted_objects):
            input_idx = i + 1
            inputs.extend(["-loop", "1", "-i", obj.source])

            # オブジェクトごとのフィルタチェーン構築
            obj_filters = []

            # Transform処理（posを除く）
            for t in obj.transforms:
                if t.name == "resize":
                    sx = t.params.get("sx", 1)
                    sy = t.params.get("sy", 1)
                    obj_filters.append(f"scale=iw*{sx}:ih*{sy}")

            # Effect処理（float引数はstart_time〜start_time+durationで値→1.0にアニメーション）
            dur = obj.duration or 5
            start = obj.start_time
            for e in obj.effects:
                if e.name == "scale":
                    v = e.params.get("value", 1)
                    expr = f"{v}+(1-{v})*(t-{start})/{dur}"
                    obj_filters.append(
                        f"scale=w='trunc(iw*({expr})/2)*2':h='trunc(ih*({expr})/2)*2':eval=frame"
                    )
                elif e.name == "fade":
                    alpha = e.params.get("alpha", 1.0)
                    expr = f"{alpha}+(1-{alpha})*(T-{start})/{dur}"
                    obj_filters.append("format=rgba")
                    obj_filters.append(
                        f"geq=r='r(X\\,Y)':g='g(X\\,Y)':b='b(X\\,Y)':a='alpha(X\\,Y)*clip({expr}\\,0\\,1)'"
                    )

            # フィルタがあればラベル付きで追加
            obj_label = f"[obj{input_idx}]"
            if obj_filters:
                filter_parts.append(
                    f"[{input_idx}:v]{','.join(obj_filters)}{obj_label}"
                )
            else:
                obj_label = f"[{input_idx}:v]"

            # overlay位置の決定（Object._posから取得）
            x_expr = "(W-w)/2"
            y_expr = "(H-h)/2"
            if obj._pos is not None:
                x = obj._pos.get("x", 0.5)
                y = obj._pos.get("y", 0.5)
                anchor = obj._pos.get("anchor", "topleft")
                if anchor == "center":
                    x_expr = f"W*{x}-w/2"
                    y_expr = f"H*{y}-h/2"
                else:
                    x_expr = f"W*{x}"
                    y_expr = f"H*{y}"

            # 時間制御: enable='between(t,start,end)'
            enable_str = ""
            if obj.duration is not None:
                start = obj.start_time
                end = start + obj.duration
                enable_str = f":enable='between(t\\,{start}\\,{end})'"

            out_label = f"[v{input_idx}]"
            filter_parts.append(
                f"{current_base}{obj_label}overlay={x_expr}:{y_expr}{enable_str}{out_label}"
            )
            current_base = out_label

        cmd = ["ffmpeg", "-y"]
        cmd.extend(inputs)

        if filter_parts:
            cmd.extend(["-filter_complex", ";".join(filter_parts)])
            cmd.extend(["-map", current_base])

        cmd.extend([
            "-c:v", "libx264",
            "-t", str(self.duration),
            "-pix_fmt", "yuv420p",
            output_path,
        ])

        return cmd


class TransformChain:
    """複数のTransformをまとめたチェーン"""
    def __init__(self, transforms):
        self.transforms = list(transforms)

    def __or__(self, other):
        """TransformChain | Transform/TransformChain → TransformChain"""
        if isinstance(other, Transform):
            return TransformChain(self.transforms + [other])
        if isinstance(other, TransformChain):
            return TransformChain(self.transforms + other.transforms)
        return NotImplemented

    def __repr__(self):
        return f"TransformChain({self.transforms})"


class EffectChain:
    """複数のEffectをまとめたチェーン"""
    def __init__(self, effects):
        self.effects = list(effects)

    def __and__(self, other):
        """EffectChain & Effect/EffectChain → EffectChain"""
        if isinstance(other, Effect):
            return EffectChain(self.effects + [other])
        if isinstance(other, EffectChain):
            return EffectChain(self.effects + other.effects)
        return NotImplemented

    def __repr__(self):
        return f"EffectChain({self.effects})"


class Transform:
    def __init__(self, name, **params):
        self.name = name
        self.params = params

    def __or__(self, other):
        """Transform | Transform/TransformChain → TransformChain"""
        if isinstance(other, Transform):
            return TransformChain([self, other])
        if isinstance(other, TransformChain):
            return TransformChain([self] + other.transforms)
        return NotImplemented

    def __repr__(self):
        return f"Transform({self.name}, {self.params})"


class Effect:
    def __init__(self, name, **params):
        self.name = name
        self.params = params

    def __and__(self, other):
        """Effect & Effect/EffectChain → EffectChain"""
        if isinstance(other, Effect):
            return EffectChain([self, other])
        if isinstance(other, EffectChain):
            return EffectChain([self] + other.effects)
        return NotImplemented

    def __repr__(self):
        return f"Effect({self.name}, {self.params})"


class Object:
    def __init__(self, source):
        self.source = source
        self.transforms = []  # pos以外のTransformリスト
        self.effects = []
        self.duration = None
        self.start_time = 0
        self.priority = 0
        self._pos = None  # pos Transformの params を保持
        # 現在のProjectに自動登録
        if Project._current is not None:
            Project._current.objects.append(self)

    def time(self, duration):
        """表示時間を設定"""
        self.duration = duration
        return self

    def __le__(self, rhs):
        """<= 演算子: Transform/TransformChain/Effect/EffectChainを適用"""
        if isinstance(rhs, Transform):
            self._apply_transform(rhs)
        elif isinstance(rhs, TransformChain):
            for t in rhs.transforms:
                self._apply_transform(t)
        elif isinstance(rhs, Effect):
            self.effects.append(rhs)
        elif isinstance(rhs, EffectChain):
            self.effects.extend(rhs.effects)
        else:
            raise TypeError(f"Object <= に渡せるのは Transform/TransformChain/Effect/EffectChain のみ: {type(rhs)}")
        return self

    def _apply_transform(self, t):
        """Transformを適用。posならば_posに保存、それ以外はtransformsに追加"""
        if t.name == "pos":
            self._pos = dict(t.params)
        else:
            self.transforms.append(t)

    def __repr__(self):
        return f"Object({self.source}, transforms={self.transforms}, effects={self.effects}, pos={self._pos})"


# --- Transform関数 ---

def resize(**kwargs):
    return Transform("resize", **kwargs)


def pos(**kwargs):
    return Transform("pos", **kwargs)


# --- Effect関数 ---

def scale(value):
    return Effect("scale", value=value)


def fade(**kwargs):
    return Effect("fade", **kwargs)
