import matplotlib.pyplot as plt
from parse_track import parse_track
from skidpad import Point, SkidPadPlanner
from euclid3 import Vector2


def main():
    cone_points, x, y, d = parse_track('skidpad.yaml')
    cones = [Point(p[0], p[1]) for p in cone_points]
    start_point = Point(x, y)
    start_direction = Vector2(0, 1)

    planner = SkidPadPlanner(cones, start_point, start_direction)
    x, y = planner.get_path()

    plt.plot([p[0] for p in cone_points], [p[1] for p in cone_points], 'x')
    plt.plot(x, y, '-')
    plt.axis('equal')
    plt.show()


if __name__ == '__main__':
    main()
