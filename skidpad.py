import numpy as np
import math
import matplotlib.pyplot as plt
from euclid3 import Vector2, Point2, LineSegment2
from parse_track import parse_track
from circle_fit import least_squares_circle, plot_data_circle, hyper_fit
from scipy.spatial import *
from spline import calc_spline_course


class SkidPadPlanner:

    def __init__(self, cones, start_point, start_vector, circle_radius=9.125):
        self.cones = cones
        self.start_point = start_point
        self.start_vector = start_vector.normalized()
        self.circle_radius = circle_radius

    def _sort_cones_by_start_pose(self):
        left_cones, right_cones = [], []

        for cone in self.cones:
            vector = Vector2(self.start_point.x - cone.x, self.start_point.y - cone.y).normalized()
            angle_with_start_vector = abs(math.atan2(vector.y, vector.x) - math.atan2(self.start_vector.y,
                                                                                      self.start_vector.x))
            if angle_with_start_vector < math.pi:
                left_cones.append(cone)
            else:
                right_cones.append(cone)

        return left_cones, right_cones

    def _get_middle_of_circles(self):
        left_cones, right_cones = self._sort_cones_by_start_pose()
        xl, yl, _ = circle_filtering([[p.x, p.y] for p in left_cones], 10)
        xr, yr, _ = circle_filtering([[p.x, p.y] for p in right_cones], 10)

        return Point2(xl, yl), Point2(xr, yr)

    def get_path_spline(self):
        point_left, point_right = self._get_middle_of_circles()
        center_point = Point2((point_left.x + point_right.x) / 2, (point_right.y + point_left.y) / 2)

        way_points = [(self.start_point.x, self.start_point.y)]

        right_vec = LineSegment2(point_right, center_point).v.normalized()
        angle = math.atan2(right_vec.y, right_vec.x)

        R = point_right.distance(center_point)

        theta_fit = np.linspace(angle, angle - 2 * math.pi, 10)

        x_fit = point_right.x + R * np.cos(theta_fit)
        y_fit = point_right.y + R * np.sin(theta_fit)

        way_points += list(zip(x_fit, y_fit))[:-1] + list(zip(x_fit, y_fit))[:-1]

        left_vec = LineSegment2(point_left, center_point).v.normalized()
        angle = math.atan2(left_vec.y, left_vec.x)

        R = point_left.distance(center_point)

        theta_fit = np.linspace(angle, angle + 2 * math.pi, 10)

        x_fit = point_left.x + R * np.cos(theta_fit)
        y_fit = point_left.y + R * np.sin(theta_fit)

        way_points += list(zip(x_fit, y_fit))[:-1] + list(zip(x_fit, y_fit))[:-1]
        way_points.append((0, 10))
        return [p[0] for p in way_points], [p[1] for p in way_points]


def filter_points(from_point, points, distance):
    new = []
    for p in points:
        if Point2(from_point[0], from_point[1]).distance(Point2(p[0], p[1])) <= distance:
            new.append(p)
    return new


def points_with_noise(points, possible_noise):
    import random
    new_points = []
    for p in points:
        new = []
        r = random.uniform(-possible_noise, possible_noise)
        new.append(p[0] + r)
        r = random.uniform(-possible_noise, possible_noise)
        new.append(p[1] + r)
        new_points.append(new)
    return new_points


def circle_filtering(points, iterations):
    xc, yc, R, residu = least_squares_circle(points)
    while len(points) > 3 and iterations > 1:
        xc, yc, R, residu = least_squares_circle(points)

        plot_data_circle([p[0] for p in points], [p[1] for p in points], xc, yc, R)

        n = []
        for p in points:
            if Point2(p[0], p[1]).distance(Point2(float(xc), float(yc))) <= R:
                n.append(p)
        points = n
        iterations -= 1

    return xc, yc, R


def get_avg_vector(points):
    x_sum = 0
    y_sum = 0
    for i in range(len(points) - 1):
        vec = LineSegment2(Point2(points[i][0], points[i][1]),
                           Point2(points[i + 1][0], points[i + 1][1])).v.normalized()
        x_sum += vec.x
        y_sum += vec.y

    avg_vec = Vector2(x_sum, y_sum).normalized()

    return [avg_vec.x, avg_vec.y]


def main():
    points, x, y, d = parse_track('skidpad.yaml')

    points = points_with_noise(points, 1)
    points = filter_points(Point2(x, y), points, 200)

    sp = SkidPadPlanner([Point2(p[0], p[1]) for p in points], Point2(x, y), Vector2(0, 1))
    l, r = sp._get_middle_of_circles()

    plt.plot([p[0] for p in points], [p[1] for p in points], '.')

    plt.plot(l.x, l.y, 'x')
    plt.plot(r.x, r.y, 'x')

    x, y = sp.get_path_spline()

    x, y, _, _, _ = calc_spline_course(x, y)
    plt.plot(x, y, '-')

    plt.axis('equal')
    plt.show()


if __name__ == '__main__':
    main()
