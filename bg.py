from scriptvedit import *

bg = Object("bg_pattern_ishigaki.jpg")
bg.time(6)
bg <= resize(sx=1, sy=1) | pos(x=0.5, y=0.5, anchor="center")
bg <= scale(1.5) & fade(alpha=0)
