import math
from euclid3 import Point2, LineSegment2
from scipy.spatial import Delaunay


ORIENTATION_MULTIPLIER = 10
MAX_EDGE_SIZE = 10


class ConeGraph(Delaunay):

    def __init__(self, points, start_position=None, start_orientation=None):
        self.start_position = start_position
        self.start_orientation = start_orientation
        Delaunay.__init__(self, points)
        self.start_edge, self.start_triangle = self._get_nearest_intersecting_triangle(
            start_position, start_orientation)
        
    class Cone(Point2):
        """ Represents a cone """

        def __init__(self, x, y):
            """ Initializes a cone with the given x and y coordinates and sets the cone's side (left or right cone).

            :param x: The x coordinate of the cone
            :param y: The y coordinate of the cone
            :param side: The side of the cone, one of the module constants LEFT or RIGHT
            """

            Point2.__init__(self, float(x), float(y))

    class MidPoint(Point2):
        """ Represents """

        def __init__(self, x, y):
            Point2.__init__(self, float(x), float(y))

        def __eq__(self, other):
            return abs(self.x - other.x) < 0.01 and abs(self.y - other.y) < 0.01

        def __hash__(self):
            return hash((self.x, self.y))

    def _get_nearest_intersecting_triangle(self, start_position, start_orientation):
        """
        Gets the nearest intersecting triangle, by seaching with a Line from a given point is heading
        :param start_position: An euclid3.Point2 object, indicating the current position from where the search begins
        :param start_orientation: An euclid.Vector2 object, indicating the current orientation to where it is searched
        """

        if not (start_position and start_orientation):
            return None, None

        triangles = self.simplices
        points = self.points

        start_orientation = start_orientation.normalize()
        car_line = LineSegment2(
            start_position, ORIENTATION_MULTIPLIER * start_orientation)

        closest_edge = None
        current_car_cut_distance = 10000

        closest_triangles_indices = []
        triangle_index = 0
        for triangle in triangles:

            current_edges = [(triangle[0], triangle[1]),
                             (triangle[1], triangle[2]), (triangle[0], triangle[2])]

            for edge in current_edges:
                p1 = self.Cone(points[edge[0]][0], points[edge[0]][1])
                p2 = self.Cone(points[edge[1]][0], points[edge[1]][1])
                edge_as_line = LineSegment2(p1, p2)

                cut = car_line.intersect(edge_as_line)
                if not cut:
                    continue

                if abs(start_position - cut) <= current_car_cut_distance or not closest_edge:
                    closest_edge = edge
                    current_car_cut_distance = abs(start_position - cut)
                    closest_triangles_indices.append(triangle_index)

            triangle_index += 1

        if closest_edge is None:
            return None, None

        for i in closest_triangles_indices:
            if closest_edge[0] in triangles[i] and closest_edge[1] in triangles[i]:

                other_point = -1
                # check if thats the looked at triangle
                for point in triangles[i]:
                    if point != closest_edge[0] and point != closest_edge[1]:
                        other_point = point

                long_car_line = LineSegment2(
                    start_position, ORIENTATION_MULTIPLIER * start_orientation * 5)

                cone1 = self.Cone(
                    x=points[closest_edge[0]][0], y=points[closest_edge[0]][1])
                cone2 = self.Cone(
                    x=points[closest_edge[1]][0], y=points[closest_edge[1]][1])
                cone3 = self.Cone(
                    x=points[other_point][0], y=points[other_point][1])

                edge1 = LineSegment2(cone1, cone3)
                edge2 = LineSegment2(cone2, cone3)

                cut1 = edge1.intersect(long_car_line)
                cut2 = edge2.intersect(long_car_line)

                if cut1 or cut2:
                    return closest_edge, i
        return None, None

    def _get_other_edges_in_triangle(self, edge, triangle):
        # get the information about the delaunay-graph
        triangles = self.simplices
        points = self.points
        neighbors = self.neighbors

        # find the point-index of the point which isn't part of the current edge
        other_point_index = -1
        for ind in triangles[triangle]:
            if ind != edge[0] and ind != edge[1]:
                other_point_index = ind

        if other_point_index == -1:
            return None, None

        # the two edge-options that are possible
        edge_option1 = (edge[0], other_point_index)
        edge_option2 = (other_point_index, edge[1])

        opposite_point_index = list(triangles[triangle]).index(edge[0])
        # index of the triangle opposite to cone1
        triangle1 = neighbors[triangle][opposite_point_index]

        opposite_point_index = list(triangles[triangle]).index(edge[1])
        # index of the triangle opposite to cone2
        triangle2 = neighbors[triangle][opposite_point_index]

        return (edge_option1, triangle2), (edge_option2, triangle1)

    def _get_middle_point_of_edge(self, edge):
        """
        Gets the middle-points of two points in the graph
        :param edge:            Edge
        :returns:               A MidPoint-object representing the middle-point
        """
        points = self.points
        point1 = points[edge[0]]
        point2 = points[edge[1]]
        return self.MidPoint((point1[0] + point2[0]) / 2, (point1[1] + point2[1]) / 2)

    def _get_edge_length(self, edge):
        points = self.points
        cone1 = self.Cone(x=points[edge[0]][0], y=points[edge[0]][1])
        cone2 = self.Cone(x=points[edge[1]][0], y=points[edge[1]][1])
        return cone1.distance(cone2)

    def _is_triangle_out_of_track(self, simplex_index):
        """ Decides, if a triangle can be a triangle inside the track, returns a boolean acoordingly

            :param simplex_index:   The index of the looked-in triangle
            :returns:               A boolean telling if the triangle is a off-track
        """
        
        if simplex_index == -1:
            return True

        points = self.points
        point_indices = self.simplices[simplex_index]
        cone1 = self.Cone(x=points[point_indices[0]][0], y=points[point_indices[0]][1])
        cone2 = self.Cone(x=points[point_indices[1]][0], y=points[point_indices[1]][1])
        cone3 = self.Cone(x=points[point_indices[2]][0], y=points[point_indices[2]][1])
        edge1 = LineSegment2(cone1, cone2)
        edge2 = LineSegment2(cone1, cone3)
        edge3 = LineSegment2(cone2, cone3)
        
        if edge1.length > MAX_EDGE_SIZE or edge2.length > MAX_EDGE_SIZE or edge3.length > MAX_EDGE_SIZE:
            return True

        if math.acos(edge1.v.normalized().dot(edge2.v.normalized())) < 0.1:
            return True
        if math.acos(edge2.v.normalized().dot(edge3.v.normalized())) < 0.1:
            return True
        if math.acos(edge2.v.normalized().dot(edge1.v.normalized())) < 0.1:
            return True

        return False

    def get_middle_path(self):
        edge = self.start_edge
        triangle = self.start_triangle
        orientation = self.start_orientation

        mid_path = []
        visited_triangles = set()
        avoided_triangles = set()

        while True:
            (edge_option1, triangle_option1), (edge_option2, triangle_option2) = self._get_other_edges_in_triangle(
                edge, triangle)

            orientation = orientation.normalized()
            vector1 = LineSegment2(self._get_middle_point_of_edge(edge),
                                   self._get_middle_point_of_edge(edge_option1)).v.normalized()
            vector2 = LineSegment2(self._get_middle_point_of_edge(edge),
                                   self._get_middle_point_of_edge(edge_option2)).v.normalized()

            angle_orientation_vector1 = abs(
                math.acos(orientation.dot(vector1))) / self._get_edge_length(edge_option1)
            angle_orientation_vector2 = abs(
                math.acos(orientation.dot(vector2))) / self._get_edge_length(edge_option2)

            if (angle_orientation_vector1 < angle_orientation_vector2 or triangle_option2 in avoided_triangles) and not self._is_triangle_out_of_track(triangle_option1):
                edge = edge_option1
                triangle = triangle_option1
                orientation = vector1
                avoided_triangles.add(triangle_option2)
            
            elif (angle_orientation_vector2 < angle_orientation_vector1 or triangle_option1 in avoided_triangles) and not self._is_triangle_out_of_track(triangle_option2):
                edge = edge_option2
                triangle = triangle_option2
                orientation = vector2
                avoided_triangles.add(triangle_option1)

            else:
                break

            new_mid_point = self._get_middle_point_of_edge(edge)

            mid_path.append((new_mid_point.x, new_mid_point.y))

            if triangle in visited_triangles or triangle == -1:
                break

            visited_triangles.add(triangle)

        return mid_path, False

    def plot(self, extra_points=None):
        if extra_points is None:
            extra_points = []
        import matplotlib.pyplot as plt

        plt.triplot(
            self.points[:, 0], self.points[:, 1], self.simplices)
        plt.plot(self.points[:, 0], self.points[:, 1], '.')
        x = [p[0] for p in extra_points]
        y = [p[1] for p in extra_points]
        plt.plot(x, y, '-', color='red')
        
        plt.gca().set_aspect('equal', adjustable='box')
        # plt.show()
