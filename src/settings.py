###########################
# DO NOT CHANGE ANYTHING. #
###########################

# ---COLORS--- #
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
DARKGRAY = (40, 40, 40)
LIGHTGRAY = (100, 100, 100)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
PURPLE = (128, 0, 128)
YELLOW = (255, 255, 0)
NUMBERCOLOR = (255, 0, 0)
BGCOLOUR = DARKGRAY

# ---GAME SETTINGS--- #
WIDTH = 1024
HEIGHT = 768
FPS = 60
TITLE = "The Harmonizer: Ultra Deluxe"
BOARDSIZE = 400
GAMESIZE = [5, 5]
TILESIZE = BOARDSIZE // max(GAMESIZE[0], GAMESIZE[1])

# ---LEVELS--- #
# 0: Empty
# 1: Red
# 2: Blue
# 3: Purple
# 4: Wall
# 5: Conveyor Down
# 6: Conveyor Up
# 7: Conveyor Right
# 8: Conveyor Left
LEVELS = [["04442",
           "04000",
           "88840",
           "04040",
           "10040"],
          ["175444",
           "045580",
           "405500",
           "468540",
           "000040",
           "040042"],
          ["4000004",
           "0004000",
           "0458640",
           "0054600",
           "0477640",
           "0244410",
           "4400044"]]

# ---STARTING POSITION--- #
START = [(WIDTH - BOARDSIZE) // 2, (HEIGHT - BOARDSIZE) // 4]
