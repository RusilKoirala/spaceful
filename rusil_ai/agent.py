
from settings import RAY_LENGTH, MAX_SPEED


def drive(car, track):
    rays = car.get_distances(track)
    inputs = [ d/ RAY_LENGTH for d in rays ]
    inputs.append(car.speed / MAX_SPEED)

    forward, backward, left, right = car.brain.feed_forward(inputs)

    if left:
        car.rotate(left=True)
    if right:
        car.rotate(right=True)

    if forward:
        car.move_forward()
    elif backward:
        car.move_backward()
    else:
        car.reduce_speed()
