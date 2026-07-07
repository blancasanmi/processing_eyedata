
# to check: it should be the computer's lab pixel sixes
SCREEN_W = 1920
SCREEN_H = 1080

# this comes from the experiment: very important to double check and make sure it is alright
FONT_SIZE = 40        # px
LINE_HEIGHT = 2.2     # unitless
LETTER_SPACING = 2    # in px as well
GAP = 40              # px, between flex children (x2 because hr counts as a child)
TEXT_MAX_W = SCREEN_W * 0.90  # also from the experiment

CHAR_W = FONT_SIZE * 0.6 + LETTER_SPACING   # ≈ 26px per character
LINE_H = FONT_SIZE * LINE_HEIGHT  # 88px from the def in https://developer.mozilla.org/en-US/docs/Web/CSS/Reference/Properties/line-height#:~:text=multiplied%20by%20the%20element%27s%20own%20font%20size 

