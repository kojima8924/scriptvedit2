from scriptvedit import *

# プリセット定義
preset_t = resize(sx=0.3, sy=0.3) | pos(x=0.3, y=0.7, anchor="center")
preset_e = scale(0.5) & fade(alpha=0)

onigiri = Object("onigiri_tenmusu.png")
onigiri | preset_t
onigiri.time(6) & preset_e
