WIDTH, HEIGHT = 1280,720
FPS = 60


ACCELERATION = 0.08
FRICTION = 0.04
ROTATION_SPEED = 1.2
MAX_SPEED=2

CAR_SIZE = (20, 10)


START_GRID = [
    (585, 591, 0),
    (560, 609, 0),
    (535, 591, 0),
    (510, 609, 0),
    (485, 591, 0),
    (460, 609, 0),
]
OFFSETS = [90,60,30,0,-30,-60,-90]
RAY_LENGTH = 200
RAY_STEP = 2


# Kill a car that stays slower than STALL_SPEED for STALL_FRAMES frames
STALL_SPEED = 0.1
STALL_FRAMES = 180
# Hard limit on how long one round can last
ROUND_FRAMES = FPS * 60


TEST = True 