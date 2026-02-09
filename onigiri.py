from scriptvedit import *

onigiri = Object("onigiri_tenmusu.png")
onigiri <= resize(sx=0.3, sy=0.3) | pos(x=0.3, y=0.7, anchor="center")
onigiri.time(6) <= scale(0.5) & fade(alpha=0)
