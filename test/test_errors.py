# エラーケーステスト: 各種エラー条件の自動検証
import sys, os, tempfile
sys.path.insert(0, "..")
from scriptvedit import (
    _resolve_param, Project, P, Object, VideoView, AudioView,
    again, move, fade, resize, AudioEffect, AudioEffectChain,
    subtitle, bubble, diagram, circle, label,
    Transform, TransformChain, Effect, EffectChain,
)


def test_math_sin_in_lambda():
    """lambda内でmath.sinを使用 → TypeErrorかつ案内メッセージ"""
    import math
    try:
        _resolve_param(lambda u: math.sin(u))
        return False, "例外が発生しませんでした"
    except TypeError as e:
        msg = str(e)
        if "scriptvedit" in msg and "sin" in msg:
            return True, msg.split('\n')[0]
        return False, f"メッセージが不適切: {msg}"


def test_undefined_anchor():
    """未定義アンカー参照 → RuntimeError"""
    layer_code = (
        'from scriptvedit import *\n'
        'pause.until("nonexistent")\n'
        'obj = Object("../onigiri_tenmusu.png")\n'
        'obj.time(1) <= move(x=0.5, y=0.5, anchor="center")\n'
    )
    temp_path = os.path.join(os.path.dirname(__file__), "_tmp_err_undef.py")
    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            f.write(layer_code)
        p = Project()
        p.configure(width=320, height=240, fps=1, background_color="black")
        p.layer(temp_path, priority=0)
        p.render("_tmp_err.mp4", dry_run=True)
        return False, "例外が発生しませんでした"
    except RuntimeError as e:
        msg = str(e)
        if "nonexistent" in msg and "pause.until" in msg:
            return True, msg.split('\n')[0]
        return False, f"メッセージが不適切: {msg}"
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_same_anchor_different_files():
    """異ファイル間で同名アンカー定義 → RuntimeError"""
    layer1_code = (
        'from scriptvedit import *\n'
        'obj = Object("../onigiri_tenmusu.png")\n'
        'obj.time(1) <= move(x=0.5, y=0.5, anchor="center")\n'
        'anchor("my_anchor")\n'
    )
    layer2_code = (
        'from scriptvedit import *\n'
        'anchor("my_anchor")\n'
        'obj = Object("../onigiri_tenmusu.png")\n'
        'obj.time(1) <= move(x=0.5, y=0.5, anchor="center")\n'
    )
    temp1 = os.path.join(os.path.dirname(__file__), "_tmp_err_dup1.py")
    temp2 = os.path.join(os.path.dirname(__file__), "_tmp_err_dup2.py")
    try:
        with open(temp1, "w", encoding="utf-8") as f:
            f.write(layer1_code)
        with open(temp2, "w", encoding="utf-8") as f:
            f.write(layer2_code)
        p = Project()
        p.configure(width=320, height=240, fps=1, background_color="black")
        p.layer(temp1, priority=0)
        p.layer(temp2, priority=1)
        p.render("_tmp_err.mp4", dry_run=True)
        return False, "例外が発生しませんでした"
    except RuntimeError as e:
        msg = str(e)
        if "my_anchor" in msg and "再定義は禁止" in msg:
            return True, msg.split('\n')[0]
        return False, f"メッセージが不適切: {msg}"
    finally:
        for f in [temp1, temp2]:
            if os.path.exists(f):
                os.unlink(f)


def test_configure_typo():
    """configure()のtypo → ValueError"""
    p = Project()
    try:
        p.configure(widht=1280)
        return False, "例外が発生しませんでした"
    except ValueError as e:
        msg = str(e)
        if "widht" in msg and "width" in msg:
            return True, msg
        return False, f"メッセージが不適切: {msg}"


def test_percent_value():
    """50%P == 0.5 の確認"""
    result = 50%P
    if result == 0.5:
        return True, f"50%P = {result}"
    return False, f"50%P = {result} (期待: 0.5)"


def test_cache_invalid():
    """cache='invalid' → ValueError"""
    p = Project()
    try:
        p.layer("dummy.py", cache="invalid")
        return False, "例外が発生しませんでした"
    except ValueError as e:
        msg = str(e)
        if "invalid" in msg:
            return True, msg
        return False, f"メッセージが不適切: {msg}"


def test_cache_use_no_file():
    """cache='use' でファイル不在 → FileNotFoundError"""
    p = Project()
    p.configure(width=320, height=240, fps=1, background_color="black")
    p.layer("nonexistent_layer.py", cache="use")
    try:
        p.render("_tmp.mp4", dry_run=True)
        return False, "例外が発生しませんでした"
    except FileNotFoundError as e:
        msg = str(e)
        if "キャッシュファイルが見つかりません" in msg:
            return True, msg.split('\n')[0]
        return False, f"メッセージが不適切: {msg}"


def test_image_length():
    """画像の length() → TypeError"""
    p = Project()
    obj = Object("../onigiri_tenmusu.png")
    try:
        obj.length()
        return False, "例外が発生しませんでした"
    except TypeError as e:
        msg = str(e)
        if "画像" in msg:
            return True, msg
        return False, f"メッセージが不適切: {msg}"


def test_missing_file_length():
    """存在しないファイルの length() → FileNotFoundError"""
    p = Project()
    obj = Object("nonexistent_video.mp4")
    try:
        obj.length()
        return False, "例外が発生しませんでした"
    except FileNotFoundError as e:
        msg = str(e)
        if "メディアの長さを取得できません" in msg:
            return True, msg
        return False, f"メッセージが不適切: {msg}"


def test_view_time_forbidden():
    """VideoView.time() / AudioView.time() → TypeError"""
    p = Project()
    obj = Object("../onigiri_tenmusu.png")
    vv = VideoView(obj)
    try:
        vv.time(3)
        return False, "VideoView.time() 例外が発生しませんでした"
    except TypeError as e:
        msg = str(e)
        if "禁止" in msg:
            return True, msg
        return False, f"メッセージが不適切: {msg}"


def test_view_until_forbidden():
    """VideoView.until() / AudioView.until() → TypeError"""
    p = Project()
    obj = Object("../Impact-38.mp3")
    av = AudioView(obj)
    try:
        av.until("test")
        return False, "AudioView.until() 例外が発生しませんでした"
    except TypeError as e:
        msg = str(e)
        if "禁止" in msg:
            return True, msg
        return False, f"メッセージが不適切: {msg}"


def test_video_audio_effect_mismatch():
    """VideoView <= again() → TypeError"""
    p = Project()
    obj = Object("../onigiri_tenmusu.png")
    vv = VideoView(obj)
    try:
        vv <= again(0.5)
        return False, "例外が発生しませんでした"
    except TypeError as e:
        msg = str(e)
        if "映像系のみ" in msg:
            return True, msg
        return False, f"メッセージが不適切: {msg}"


def test_audio_video_effect_mismatch():
    """AudioView <= move() → TypeError"""
    p = Project()
    obj = Object("../Impact-38.mp3")
    av = AudioView(obj)
    try:
        av <= move(x=0.5, y=0.5)
        return False, "例外が発生しませんでした"
    except TypeError as e:
        msg = str(e)
        if "音声系のみ" in msg:
            return True, msg
        return False, f"メッセージが不適切: {msg}"


def test_web_kwargs_on_non_web():
    """画像にduration/sizeを渡す → TypeError"""
    p = Project()
    try:
        Object("../onigiri_tenmusu.png", duration=2.0, size=(640, 360))
        return False, "例外が発生しませんでした"
    except TypeError as e:
        msg = str(e)
        if "web Object" in msg and ".html" in msg:
            return True, msg
        return False, f"メッセージが不適切: {msg}"


def test_web_unknown_kwarg():
    """HTMLに不明なkwarg → TypeError"""
    p = Project()
    try:
        Object("test.html", duration=2.0, size=(640, 360), unknown_param=True)
        return False, "例外が発生しませんでした"
    except TypeError as e:
        msg = str(e)
        if "不明なキーワード引数" in msg and "unknown_param" in msg:
            return True, msg
        return False, f"メッセージが不適切: {msg}"


def test_web_no_duration():
    """HTMLにdurationなし → ValueError"""
    p = Project()
    try:
        Object("test.html", size=(640, 360))
        return False, "例外が発生しませんでした"
    except ValueError as e:
        msg = str(e)
        if "duration" in msg and "必須" in msg:
            return True, msg
        return False, f"メッセージが不適切: {msg}"


def test_subtitle_no_project():
    """subtitle() でProject未設定 + size省略 → RuntimeError"""
    old = Project._current
    Project._current = None
    try:
        subtitle("テスト")
        return False, "例外が発生しませんでした"
    except RuntimeError as e:
        msg = str(e)
        if "アクティブなProject" in msg:
            return True, msg
        return False, f"メッセージが不適切: {msg}"
    finally:
        Project._current = old


def test_diagram_no_project():
    """diagram() でProject未設定 + size省略 → RuntimeError"""
    old = Project._current
    Project._current = None
    try:
        diagram([circle(0.5, 0.5, 0.1)])
        return False, "例外が発生しませんでした"
    except RuntimeError as e:
        msg = str(e)
        if "アクティブなProject" in msg:
            return True, msg
        return False, f"メッセージが不適切: {msg}"
    finally:
        Project._current = old


def test_subtitle_with_explicit_size():
    """subtitle() にsize明示 → Project不要で成功"""
    old = Project._current
    Project._current = None
    try:
        obj = subtitle("テスト", size=(640, 360))
        if obj.media_type == "web" and obj._web_size == (640, 360):
            return True, f"size=(640,360) で正常生成"
        return False, f"属性が不正: type={obj.media_type}, size={obj._web_size}"
    finally:
        Project._current = old


def test_neg_transform():
    """-resize() → policy='off' のTransformを返す"""
    result = -resize(sx=0.5, sy=0.5)
    if not isinstance(result, Transform):
        return False, f"型が不正: {type(result)}"
    if result.policy != "off":
        return False, f"policyが不正: {result.policy}"
    return True, f"policy={result.policy}"


def test_neg_effect():
    """-scale(0.5) → policy='off' のEffectを返す"""
    from scriptvedit import scale
    result = -scale(0.5)
    if not isinstance(result, Effect):
        return False, f"型が不正: {type(result)}"
    if result.policy != "off":
        return False, f"policyが不正: {result.policy}"
    return True, f"policy={result.policy}"


def test_chain_sugar():
    """~(tf1 | tf2) で全opがquality='fast'になる"""
    tf1 = resize(sx=0.5, sy=0.5)
    tf2 = resize(sx=0.3, sy=0.3)
    chain = tf1 | tf2
    result = ~chain
    if not isinstance(result, TransformChain):
        return False, f"型が不正: {type(result)}"
    # 全opがquality="fast"
    for i, t in enumerate(result.transforms):
        if not isinstance(t, Transform):
            return False, f"transforms[{i}]がTransformでない: {type(t)}"
        if t.quality != "fast":
            return False, f"transforms[{i}].quality={t.quality} (期待: fast)"
    return True, f"全{len(result.transforms)}opがquality=fast"


def test_force_operator():
    """+op で policy='force' 確認"""
    result = +resize(sx=0.5, sy=0.5)
    if not isinstance(result, Transform):
        return False, f"型が不正: {type(result)}"
    if result.policy != "force":
        return False, f"policyが不正: {result.policy}"
    return True, f"policy={result.policy}"


def test_off_operator():
    """-op で policy='off' 確認"""
    result = -resize(sx=0.5, sy=0.5)
    if not isinstance(result, Transform):
        return False, f"型が不正: {type(result)}"
    if result.policy != "off":
        return False, f"policyが不正: {result.policy}"
    return True, f"policy={result.policy}"


def test_fast_quality():
    """~op で quality='fast' 確認"""
    result = ~resize(sx=0.5, sy=0.5)
    if not isinstance(result, Transform):
        return False, f"型が不正: {type(result)}"
    if result.quality != "fast":
        return False, f"qualityが不正: {result.quality}"
    return True, f"quality={result.quality}"


def test_chain_force():
    """+chain で末尾policy='force' 確認"""
    tf1 = resize(sx=0.5, sy=0.5)
    tf2 = resize(sx=0.3, sy=0.3)
    chain = tf1 | tf2
    result = +chain
    if not isinstance(result, TransformChain):
        return False, f"型が不正: {type(result)}"
    last = result.transforms[-1]
    if last.policy != "force":
        return False, f"末尾policy={last.policy} (期待: force)"
    first = result.transforms[0]
    if first.policy != "auto":
        return False, f"先頭policy={first.policy} (期待: auto)"
    return True, f"末尾policy=force, 先頭policy=auto"


ALL_TESTS = [
    ("math.sin in lambda", test_math_sin_in_lambda),
    ("未定義アンカー参照", test_undefined_anchor),
    ("同名アンカー異ファイル", test_same_anchor_different_files),
    ("configure typo", test_configure_typo),
    ("50%P == 0.5", test_percent_value),
    ("cache='invalid'", test_cache_invalid),
    ("cache='use' ファイル不在", test_cache_use_no_file),
    ("画像のlength()", test_image_length),
    ("存在しないファイルのlength()", test_missing_file_length),
    ("VideoView.time()禁止", test_view_time_forbidden),
    ("AudioView.until()禁止", test_view_until_forbidden),
    ("VideoView<=音声エフェクト", test_video_audio_effect_mismatch),
    ("AudioView<=映像エフェクト", test_audio_video_effect_mismatch),
    ("非webにkwargs", test_web_kwargs_on_non_web),
    ("web不明kwarg", test_web_unknown_kwarg),
    ("web duration未指定", test_web_no_duration),
    ("subtitle Project未設定", test_subtitle_no_project),
    ("diagram Project未設定", test_diagram_no_project),
    ("subtitle size明示", test_subtitle_with_explicit_size),
    ("-Transform", test_neg_transform),
    ("-Effect", test_neg_effect),
    ("~chain糖衣", test_chain_sugar),
    ("+force演算子", test_force_operator),
    ("-off演算子", test_off_operator),
    ("~fast品質", test_fast_quality),
    ("+chain force", test_chain_force),
]


if __name__ == "__main__":
    print("エラーケーステスト")
    passed = 0
    failed = 0
    for name, fn in ALL_TESTS:
        ok, msg = fn()
        status = "OK" if ok else "FAIL"
        print(f"  {name}: {status} - {msg[:80]}")
        if ok:
            passed += 1
        else:
            failed += 1
    print(f"\n結果: {passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
