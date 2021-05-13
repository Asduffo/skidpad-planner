import numpy as np
import math
from euclid3 import Point2, LineSegment2
from circle_fit import least_squares_circle
from spline import calc_spline_course


class Point(Point2):

    def __init__(self, x, y):
        """ Initializes a point with the given x and y coordinates.

        :param x: The x coordinate of the cone
        :param y: The y coordinate of the cone
        """
        Point2.__init__(self, float(x), float(y))


class SkidPadPlanner:

    def __init__(self, cones, start_point, start_vector, circle_radius=9.125):
        self.cones = cones
        self.start_point = start_point
        self.start_vector = start_vector.normalized()
        self.circle_radius = circle_radius

    def _sort_cones_by_start_pose(self):
        left_cones, right_cones = [], []

        for cone in self.cones:
            p1 = self.start_point
            p2 = LineSegment2(self.start_point, self.start_vector).p2
            det = (cone.x - p1.x) * (p2.y - p1.y) - (cone.y - p1.y) * (p2.x - p1.x)

            if det < 0:

                left_cones.append(cone)
            else:
                right_cones.append(cone)

        return left_cones, right_cones

    def _get_middle_of_circles(self):
        left_cones, right_cones = self._sort_cones_by_start_pose()

        if not (len(left_cones) > 7 and len(right_cones) > 7):
            return None, None
        xl, yl, _ = circle_filtering([[p.x, p.y] for p in left_cones], 10)
        xr, yr, _ = circle_filtering([[p.x, p.y] for p in right_cones], 10)

        return Point(xl, yl), Point(xr, yr)

    def get_path(self):
        point_left, point_right = self._get_middle_of_circles()

        center_point = Point((point_left.x + point_right.x) / 2, (point_right.y + point_left.y) / 2)
        way_points = [(self.start_point.x, self.start_point.y)]

        right_vec = LineSegment2(point_right, center_point).v.normalized()
        angle = math.atan2(right_vec.y, right_vec.x)

        R = point_right.distance(center_point)

        theta_fit = np.linspace(angle, angle - 2 * math.pi, 20)

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

        last_point = LineSegment2(self.start_point, self.start_vector * 30).p2
        way_points.append((last_point[0], last_point[1]))

        x, y, _, _, _ = calc_spline_course([p[0] for p in way_points], [p[1] for p in way_points])
        return x, y


def circle_filtering(points, max_iterations):
    xc, yc, R, _ = least_squares_circle(points)
    while len(points) > 3 and max_iterations > 1:
        xc, yc, R, _ = least_squares_circle(points)

        n = []
        for p in points:
            if Point(p[0], p[1]).distance(Point(float(xc), float(yc))) <= R:
                n.append(p)
        points = n
        max_iterations -= 1

    return xc, yc, R
