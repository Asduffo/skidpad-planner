# from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from circle_fit import hyper_fit, least_squares_circle
import numpy as np
import math
from scipy.spatial import distance
from skidpad_helpers import generate_data, rotate_cones, get_rotational_matrix, translate_cones, random_translation, \
    random_rotation
import time
import yaml


class SkidpadPlanner:

    def __init__(self, car_orientation, car_position):
        self.simulate_range = False
        self.plotting = False
        self.simulate_noise = True
        self.z = None
        self.center_left = None
        self.center_right = None
        self.radius_left = None
        self.radius_right = None
        self.track_width = 3.0
        self.cones_positions = None
        self.left_head = None
        self.right_head = None
        self.skidpad_angle = None
        self.translation = None
        # save the length of the start, finish and circle path as a class attribute for the yaw angle computation
        self.start_length = None
        self.finish_length = None
        self.circle_length = None
        self.skidpath_path = None  # full skidpad path
        self.car_orientation = car_orientation
        self.car_position = car_position
        # read the params from the yaml file
        # with open(r'skidpad_params.yaml') as file:
        #     self.args = yaml.full_load(file)

    def split_cones(self, unsorted_cones, car_orientation):
        '''
        splits a point-cloud into left and right cones w.r.t car orientation
        :param car_orientation: np.array, orientation of the car in x-,y- coordinates
        :return: left_cones, right_cones (both n,2 np.arrays)
        ## deprecated function ### doesn't work properly
        '''

        # compute side lengths of triangle
        y_axis = np.array((0.0, 1.0))
        y_length = np.linalg.norm(y_axis)
        orientation_length = np.linalg.norm(car_orientation)
        third_site = np.subtract(y_axis, car_orientation)
        third_site_length = np.linalg.norm(third_site)

        # compute numerator and denominator of cosine sentence
        numerator = third_site_length ** 2 - y_length ** 2 - orientation_length ** 2
        denominator = (-2) * orientation_length * y_length

        # compute skidpad angle
        skidpad_angle = math.degrees(math.acos(numerator / denominator))
        print("This is the estimated Skidpad angle: ", skidpad_angle)

        # print("This is the estimated Skidpad angle: ", skidpad_angle)

        left_side = []
        right_side = []
        cone_angle_list = []
        # sort cones based on estimated skidpad angle
        for current_cone in unsorted_cones:
            current_cone_side = np.linalg.norm(current_cone)
            third_site = np.subtract(current_cone, y_axis)
            third_site_length = np.linalg.norm(third_site)
            # compute numerator and denominator of cosine sentence
            numerator = third_site_length ** 2 - current_cone_side ** 2 - y_length ** 2
            denominator = (-2) * y_length * current_cone_side
            cone_angle = math.degrees(math.acos(numerator / denominator))
            cone_angle_list.append(cone_angle)
            if self.right_head:
                if cone_angle < skidpad_angle:
                    left_side.append(current_cone)
                else:
                    right_side.append(current_cone)

            else:
                if cone_angle > skidpad_angle:
                    left_side.append(current_cone)
                else:
                    right_side.append(current_cone)

            debug_on = False
            # only record angles if debuggin is wanted
            if debug_on:
                fig, ax = plt.subplots(figsize=[10.0, 10.0])
                ax.scatter(unsorted_cones[:, 0], unsorted_cones[:, 1])

                for i, txt in enumerate(cone_angle_list):
                    ax.annotate(int(txt), (unsorted_cones[i, 0], unsorted_cones[i, 1]))
                plt.grid()
                plt.savefig("debug_split.png")
                plt.close()

        return np.array(left_side), np.array(right_side), skidpad_angle

    def get_angle(self, first_vector, second_vector):
        '''
        computes the angle between two vectors based on the cosine sentence
        :param first_vector: first vector as np.array(x_component, y_component)
        :param second_vector: second vector as np.array(x_component, y_component)
        :return: angle in radians
        '''
        # compute the length of all three vectors
        first_vector_length = np.linalg.norm(first_vector)
        second_vector_length = np.linalg.norm(second_vector)
        third_vector = np.subtract(first_vector, second_vector)
        third_vector_length = np.linalg.norm(third_vector)

        # compute numerator and denominator of cosine sentence
        numerator = third_vector_length ** 2 - first_vector_length ** 2 - second_vector_length ** 2
        denominator = (-2) * second_vector_length * first_vector_length

        # return angle
        return math.acos(numerator / denominator)

    def calc_yaw_angle(self):
        '''
        computing the corresponding yaw angle for the whole skidpad path since this is also required for control
        use class attributes as inputs
        :return: list of yaw angles in radians
        '''
        # get first yaw angle based on initial car orientation and first two way points
        current_path_point = 0  # scalar that counts at which point of the overall skidpad path we currently are

        start_heading = np.subtract(self.skidpad_path[1], self.skidpad_path[0])
        yaw_angle_list = np.array((self.get_angle(car_orientation, start_heading)))
        yaw_angle_list = np.append(yaw_angle_list, np.zeros((self.start_length - 1)))

        heading_before_next_step = np.subtract(self.skidpad_path[self.start_length - 1],
                                               self.skidpad_path[self.start_length - 2])

        current_path_point += self.start_length  # update

        # print(yaw_angle_list.shape)

        # for loop to get all the yaw angles for both circles
        for number_of_steps in [self.circle_length, self.circle_length]:
            circle_heading = np.subtract(self.skidpad_path[current_path_point + 1],
                                         self.skidpad_path[current_path_point])
            yaw_angle_list = np.append(yaw_angle_list, self.get_angle(heading_before_next_step, circle_heading))
            heading_first_part = np.subtract(self.skidpad_path[current_path_point + 5], self.skidpad_path[
                current_path_point + 4])  # since yaw is constant in the circle we can take any
            heading_second_part = np.subtract(self.skidpad_path[current_path_point + 6],
                                              self.skidpad_path[current_path_point + 5])
            yaw_angle = self.get_angle(heading_first_part, heading_second_part)
            inter_yaw_list = np.repeat(yaw_angle,
                                       2 * self.circle_length - 1)  # each circle must be considered two times
            yaw_angle_list = np.append(yaw_angle_list, inter_yaw_list)
            current_path_point += 2 * number_of_steps  # each circle is driven two times
            heading_before_next_step = np.subtract(self.skidpad_path[current_path_point - 2],
                                                   self.skidpad_path[current_path_point - 1])
            # print(yaw_angle_list.shape)

        # get the yaw_angle for the last part until the car stops
        heading_after_last_circle = np.subtract(self.skidpad_path[current_path_point],
                                                self.skidpad_path[current_path_point - 1])
        yaw_angle_list = np.append(yaw_angle_list, self.get_angle(heading_before_next_step, heading_after_last_circle))
        # print(yaw_angle_list.shape)
        yaw_angle_list = np.append(yaw_angle_list, np.zeros((self.finish_length - 1)))
        print(yaw_angle_list)

        # check if each path point has a corresponding yaw angle
        if yaw_angle_list.shape[0] != self.skidpad_path.shape[0]:
            raise ValueError(
                "Yaw Angle List ({} Points) and Skidpad Path ({} Points) doesn't match".format(yaw_angle_list.shape[0],
                                                                                               self.skidpad_path.shape[
                                                                                                   0]))

        return yaw_angle_list

    def sort_cones(self, unsorted_cones, car_position, car_orientation):
        '''
        splits a point-cloud into left and right cones w.r.t car orientation
        :param car_orientation: np.array, orientation of the car in x-,y- coordinates
        :return: left_cones, right_cones (both n,2 np.arrays)
        '''
        debug_on = False

        y_axis = np.array((0.0, 1.0))

        # compute skidpad angle
        skidpad_angle = self.get_angle(y_axis, car_orientation)  # all functions below work with radians
        self.skidpad_angle = skidpad_angle
        print("This is the estimated Skidpad angle: ", math.degrees(skidpad_angle))

        # in case of left-leading switch sign of skidpad angle
        if not self.right_head:
            skidpad_angle_backward = (-1) * skidpad_angle
        else:
            skidpad_angle_backward = skidpad_angle

        if debug_on:
            fig = plt.figure(figsize=[10.0, 10.0])
            plt.scatter(unsorted_cones[:, 0], unsorted_cones[:, 1])
            plt.grid()
            plt.show()

        # back rotation of all points
        first_rot_matrix = get_rotational_matrix(skidpad_angle_backward)
        unsorted_cones = rotate_cones(first_rot_matrix, unsorted_cones)
        car_position = np.matmul(first_rot_matrix, car_position)
        if debug_on:
            fig = plt.figure(figsize=[10.0, 10.0])
            plt.scatter(unsorted_cones[:, 0], unsorted_cones[:, 1])
            plt.scatter(car_position[0], car_position[1])
            plt.grid()
            plt.show()

        #   compute necessary translation
        # translate all cones
        # inverted_translation = np.array((car_position[0]*(-1), car_position[1]*(-1)))
        # unsorted_cones = translate_cones(unsorted_cones, inverted_translation)
        if debug_on:
            fig = plt.figure(figsize=[10.0, 10.0])
            plt.scatter(unsorted_cones[:, 0], unsorted_cones[:, 1])
            plt.scatter(car_position[0], car_position[1])
            plt.grid()
            plt.show()

        # split cones by x-sign
        left_cones = []
        right_cones = []
        for current_cone in unsorted_cones:
            if current_cone[0] > car_position[0]:
                right_cones.append(current_cone)
            else:
                left_cones.append(current_cone)

        left_cones = np.array(left_cones)
        right_cones = np.array(right_cones)

        if debug_on:
            fig = plt.figure(figsize=[10.0, 10.0])
            plt.scatter(left_cones[:, 0], left_cones[:, 1])
            plt.scatter(right_cones[:, 0], right_cones[:, 1])
            plt.grid()
            plt.show()

        # transform all cones back
        if self.right_head:
            skidpad_angle_forward = (-1) * skidpad_angle
        else:
            skidpad_angle_forward = skidpad_angle

        second_rot_matrix = get_rotational_matrix(skidpad_angle_forward)

        # translate cones back
        # left_cones = translate_cones(left_cones, car_position)
        # right_cones = translate_cones(right_cones, car_position)
        # actual transformation
        left_cones = rotate_cones(second_rot_matrix, left_cones)
        right_cones = rotate_cones(second_rot_matrix, right_cones)

        if debug_on:
            fig = plt.figure(figsize=[10.0, 10.0])
            plt.scatter(left_cones[:, 0], left_cones[:, 1])
            plt.scatter(right_cones[:, 0], right_cones[:, 1])
            plt.title("Final backtransform")
            plt.grid()
            plt.show()

        return left_cones, right_cones, skidpad_angle

    def check_estimate(self, radius, fit_error):
        '''
        function check if the circle fitting is accurate enough; if not wait for next perception and fit again
        must be checked for both left and right circle
        :param radius: radius coming from circle fitting
        :param fit_error: fitting error coming from circle fitting
        :return: boolean flag if fitting was valid or not
        '''
        print("Radius: ", radius)
        print("Fitting error: ", fit_error)
        if radius > 100 or radius < 0:
            return False
        elif fit_error > 1000:
            return False
        else:
            return True

    def sort_by_distance(self, point_list, center_estimate, threshold):
        '''

        :param point_list: n,2-shape np.array containing the list of cones
        :param center_estimate: estimate of the circle center for outer and inner circle
        :return: list of points from the inner circle

        '''
        result = []
        for coordinate in point_list:
            distance = np.linalg.norm(center_estimate - coordinate)
            # print("This is the current distance:", distance)
            if distance < threshold:
                result.append(coordinate)

        return np.array(result)

    def add_noise(self, coordinates, range):
        '''

        :param coordinates: initial coordinates of the point that will be pertubated by noise
        :param range: range of the noise that will be added to the coordinate
        :return: noise-pertubated coordinates
        '''
        direction_int_x = np.random.randint(2)
        direction_int_y = np.random.randint(2)
        if direction_int_x == 0:  # pertubate in negative direction
            value_x = coordinates[0]
            noise = np.random.uniform(0.0, range)
            value_x -= noise
        else:  # pertubate in positive direction
            value_x = coordinates[0]
            noise = np.random.uniform(0.0, range)
            value_x += noise

        if direction_int_y == 0:  # pertubate in negative direction
            value_y = coordinates[1]
            noise = np.random.uniform(0.0, range)
            value_y -= noise
        else:  # pertubate in positive direction
            value_y = coordinates[1]
            noise = np.random.uniform(0.0, range)
            value_y += noise

        return np.array((value_x, value_y))

    def fit_circles(self, cone_data):
        '''
        Performs two-step fitting of a circle to the given cones position using least squares optimization
        :param cone_data: left or right side cones
        :return: circle center, circle radius
        '''
        debug_on = False
        # first fitting iteration
        center_x, center_y, radius, _ = least_squares_circle(cone_data)
        center_estimate = np.array((center_x, center_y))
        if debug_on:
            plt.scatter(cone_data[:, 0], cone_data[:, 1])
            plt.scatter(center_x, center_y)

            print("First fitted radius: ", radius)

        # second fitting iteration
        left_half_filtered = self.sort_by_distance(self.cone_positions, center_estimate, radius)
        filtered_x, filtered_y, filtered_r, fitting_error = least_squares_circle(
            left_half_filtered)  # run actucal fitting

        return filtered_x, filtered_y, filtered_r, fitting_error

    def run_skidpad(self, noise_range, unsorted_cones, car_orientation, starting_point, translation, center_skidpad,
                    end_point):
        '''

        :param noise_range: maximum noise that might be added to datapoints if simulate_noise param is enabled
        :return: radius of inner circle, centerpoint of extracted circle
        '''
        if car_orientation[0] > 0:
            self.right_head = True
        else:
            self.left_head = True

        self.cone_positions = unsorted_cones

        # simulate limited perception
        if self.simulate_range:
            self.cone_positions = self.sort_by_distance(self.cone_positions, starting_point,
                                                        10000)

        # if perception should be pertubated by noise
        if self.simulate_noise:
            left_side_new = []
            for point in self.cone_positions:
                new_point = self.add_noise(point, noise_range)
                left_side_new.append(new_point)
            self.cone_positions = np.array(left_side_new)

        # split cones into left and right circle
        left_side, right_side, skidpad_angle = self.sort_cones(self.cone_positions, starting_point, car_orientation)

        # check for false positives that may degrade the accuracy of the circle fitting
        left_side, right_side = self.check_cones(starting_point, left_side, right_side, 1000,
                                                 skidpad_angle, translation)

        # fit left and right circle
        left_x, left_y, left_r, left_error = self.fit_circles(left_side)
        right_x, right_y, right_r, right_error = self.fit_circles(right_side)
        valid_left = self.check_estimate(left_r, left_error)
        valid_right = self.check_estimate(right_r, right_error)

        if not valid_left or not valid_right:  # if circle fitting was not accurate enough no trajectory is returned
            print("Rejected circle fit due to inaccuracy")
            return None

        print("Estimated final right radius: ", right_r)
        print("Estimated final left radius: ", left_r)
        center_left = np.array((left_x, left_y))
        center_right = np.array((right_x, right_y))

        # compute path through circles with a given resolution:
        upper_left, lower_left = self.get_circle_angle_sampling(center_left, left_r, 1000)
        upper_right, lower_right = self.get_circle_angle_sampling(center_right, right_r,
                                                                  1000)

        # compute the enter and the finish line that are required to reach the circles and leave the skidpad course
        beginning = self.sample_beginning_line(starting_point, 1000,
                                               np.vstack((upper_right, lower_right)), center_skidpad)
        finish_line = self.sample_end_path(end_point, 1000,
                                           np.vstack((upper_left, lower_left)), center_skidpad)

        # set params required for yaw angle computation
        self.circle_length = upper_left.shape[0] * 2  # both circles are equal
        self.start_length = beginning.shape[0]
        self.finish_length = finish_line.shape[0]

        self.skidpad_path = self.get_skidpad_path(beginning, np.vstack((upper_left, lower_left)),
                                                  np.vstack((upper_right, lower_right)), finish_line)

        # compute corresponding yaw angle
        self.calc_yaw_angle()

        return upper_left, lower_left, upper_right, lower_right, beginning, finish_line, self.skidpad_path

    def check_cones(self, car_position, unsorted_cones_left, unsorted_cones_right, margin, skidpad_angle, translation):
        '''
        defines two circles around the centers and rejects all cones without a these two circles
        :param car_position: position of the car
        :param unsorted_cones: list of all perceived cones
        :param: margin: margin added to the exact circle radius to handle all possible inaccuracies
        :return: list of all reliable cones
        '''

        # define centers based on global coordinate system
        # current_x = car_position[0]
        # current_y = car_position[1]

        # fixed left and right center that will be later transformed
        center_left = np.array((-9.125, 0))  # 18.25 distance of the centers
        center_right = np.array((9.125, 0))

        # back_rotation = get_rotational_matrix(-math.radians(skidpad_angle))
        # car_position = np.matmul(back_rotation, car_position)

        # create rotational matrix by using the skidpad angle
        # computed skidpad angle is always positive so depending on the heading we need to switch signs here
        if self.right_head:
            rotational_matrix = get_rotational_matrix((-1) * (skidpad_angle))
        else:
            rotational_matrix = get_rotational_matrix((skidpad_angle))

        # now rotate the two centers to fit it to the actual skidpad course
        center_left = np.matmul(rotational_matrix, center_left)
        center_right = np.matmul(rotational_matrix, center_right)

        # perform translation for testing purposes only
        center_left = np.array((center_left[0] + translation[0], center_left[1] + translation[1]))
        center_right = np.array((center_right[0] + translation[0], center_right[1] + translation[1]))
        debug_on = False
        if debug_on:
            fig = plt.figure(figsize=[10.0, 10.0])

            plt.scatter(unsorted_cones_left[:, 0], unsorted_cones_left[:, 1])
            plt.scatter(unsorted_cones_right[:, 0], unsorted_cones_right[:, 1])
            plt.scatter(car_position[0], car_position[1])
            plt.scatter(center_left[0], center_left[1])
            plt.scatter(center_right[0], center_right[1])
            plt.grid()
            plt.savefig("skidpad_split.png")
            plt.close()

        list_of_valid_cones_left = []
        list_of_valid_cones_right = []

        # iterate over all cones, compute the distance to both centers and
        # we need to seperate for-loops since the number of left and right might be different
        for current_cone_left in unsorted_cones_left:
            distance_left = distance.euclidean(current_cone_left, center_left)
            if distance_left < 10.625 + margin:
                list_of_valid_cones_left.append(current_cone_left)
            else:
                print("Rejected left point")

        for current_cone_right in unsorted_cones_right:
            distance_right = distance.euclidean(current_cone_right, center_right)
            if distance_right < 10.625 + margin:
                list_of_valid_cones_right.append(current_cone_right)
            else:
                print("Rejected right point")

        return np.array(list_of_valid_cones_left), np.array(list_of_valid_cones_right)

    def sample_beginning_line(self, start, interval, right_points, skidpad_center):
        '''
        samples a straight line between two point with a given discretization
        :param start: starting point given as np.array (mostly car_position)
        :param circle_points: calc circle points of left circle
        :param interval: sample density between two points
        :return:
        '''
        distance_to_beat = 1e+8

        # find closest circle to skidpad center
        for current_cone in right_points:
            current_distance = distance.euclidean(current_cone, skidpad_center)
            if current_distance < distance_to_beat:
                distance_to_beat = current_distance
                end_cone = current_cone

        # compute points of straight line
        t_vector = np.linspace(0, 1, interval)
        b_a = np.subtract(end_cone, start)
        straight_line_points = []

        for t in t_vector:
            new_point = start + t * b_a
            straight_line_points.append(new_point)

        return np.array(straight_line_points)

    def sample_end_path(self, end, interval, left_points, skidpad_center):
        '''
        samples a straight line between two point with a given discretization
        :param start: starting point given as np.array (mostly car_position)
        :param circle_points: calc circle points of left circle
        :param interval: sample density between two points
        :return:
        '''
        distance_to_beat = 1e+8

        # find closest circle to skidpad center
        for current_cone in left_points:
            current_distance = distance.euclidean(current_cone, skidpad_center)
            if current_distance < distance_to_beat:
                distance_to_beat = current_distance
                start_cone = current_cone

        # compute points of straight line
        t_vector = np.linspace(0, 1, interval)
        b_a = np.subtract(end, start_cone)
        straight_line_points = []

        for t in t_vector:
            new_point = start_cone + t * b_a
            straight_line_points.append(new_point)

        return np.array(straight_line_points)

    def get_skidpad_path(self, beginning, left_path, right_path, end_path):
        '''
        this function takes all the seperate path fragments and put them together into one np.array
        of the form n,2
        :param beginning: enter path
        :param left_path: left circle
        :param right_path: right circle
        :param end_point: exit path
        :return: list of path point in the correct order according to the rules
        '''
        full_path = np.vstack((beginning, right_path, right_path, left_path, left_path, end_path))
        return full_path

    def get_circle_x_sampling(self):
        '''
        takes the circle center from the class as input
        samples over x-axis instead of angle
        !!Unused!!
        :return: sequence of points that are drivable
        '''

        x_points = np.linspace(0.0, (self.radius_left + (self.track_width / 2)), 50)  #
        # positive_part = np.linspace(self.center_left[0], self.center_left[0] + (self.radius_left + (self.track_width/2)), 100)

        # x_points = np.concatenate((negative_part, positive_part))

        circle_points_y_positive = []
        circle_points_y_negative = []
        for every_point in x_points:
            # print(every_point)
            r_y = math.sqrt((self.radius_left + (self.track_width / 2)) ** 2 - every_point ** 2)
            circle_points_y_positive.append(r_y)
            circle_points_y_negative.append((-1) * r_y)

        x_points_negative = [(-1) * xk for xk in x_points]

        # translate circle according to the computed center (previous computations assume zero-centred circle)
        x_points = [xz + self.center_left[0] for xz in x_points]
        x_points_negative = [xz + self.center_left[0] for xz in x_points_negative]
        circle_points_y_positive = [xz + self.center_left[1] for xz in circle_points_y_positive]
        circle_points_y_negative = [xz + self.center_left[1] for xz in circle_points_y_negative]

        quadrant_1 = np.vstack((x_points, circle_points_y_positive))
        quadrant_2 = np.vstack((x_points_negative, circle_points_y_positive))
        quadrant_2 = np.fliplr(quadrant_2)
        quadrant_1 = np.transpose(quadrant_1)
        quadrant_2 = np.transpose(quadrant_2)
        quadrant_1_2 = np.vstack((quadrant_2, quadrant_1))
        # print(quadrant_1_2)

        quadrant_3 = np.vstack((x_points_negative, circle_points_y_negative))
        quadrant_4 = np.vstack((x_points, circle_points_y_negative))
        quadrant_3 = np.fliplr(quadrant_3)
        quadrant_3 = np.transpose(quadrant_3)
        quadrant_4 = np.transpose(quadrant_4)
        quadrant_3_4 = np.vstack((quadrant_3, quadrant_4))
        # print(quadrant_3_4)

        return quadrant_1_2, quadrant_3_4

    def get_circle_angle_sampling(self, circle_center, radius, resolution):
        '''
        takes the circle center from the class as input
        :return: sequence of points that are drivable
        '''

        angle_sampling = np.linspace(0.0, math.pi / 2, resolution)  #
        # positive_part = np.linspace(self.center_left[0], self.center_left[0] + (self.radius_left + (self.track_width/2)), 100)

        # x_points = np.concatenate((negative_part, positive_part))

        circle_points_y_positive = []
        circle_points_y_negative = []
        x_points = []
        for every_angle in angle_sampling:
            # print(every_point)
            y_value = math.sin(every_angle) * (radius + self.track_width / 2)
            x_value = math.cos(every_angle) * (radius + self.track_width / 2)
            x_points.append(x_value)
            circle_points_y_positive.append(y_value)
            circle_points_y_negative.append((-1) * y_value)

        x_points_negative = [(-1) * xk for xk in x_points]

        # translate circle according to the computed center
        x_points = [xz + circle_center[0] for xz in x_points]
        x_points_negative = [xz + circle_center[0] for xz in x_points_negative]
        circle_points_y_positive = [xz + circle_center[1] for xz in circle_points_y_positive]
        circle_points_y_negative = [xz + circle_center[1] for xz in circle_points_y_negative]

        # compute point for the upper half
        quadrant_1 = np.vstack((x_points, circle_points_y_positive))
        quadrant_2 = np.vstack((x_points_negative, circle_points_y_positive))
        quadrant_2 = np.fliplr(quadrant_2)
        quadrant_1 = np.transpose(quadrant_1)
        quadrant_2 = np.transpose(quadrant_2)
        quadrant_1_2 = np.vstack((quadrant_2, quadrant_1))
        # print(quadrant_1_2)

        # compute points for the lower half
        quadrant_3 = np.vstack((x_points_negative, circle_points_y_negative))
        quadrant_4 = np.vstack((x_points, circle_points_y_negative))
        quadrant_3 = np.fliplr(quadrant_3)
        quadrant_3 = np.transpose(quadrant_3)
        quadrant_4 = np.transpose(quadrant_4)
        quadrant_3_4 = np.vstack((quadrant_3, quadrant_4))
        # print(quadrant_3_4)

        return quadrant_1_2, quadrant_3_4


if __name__ == "__main__":
    # define real data with some rotation w.r.t to global coordinate frames

    # with open(r'skidpad_params.yaml') as file:
    #     args = yaml.full_load(file)

    # define params for the test run
    rotation_angle = - 0.75 * math.pi
    # rotation_angle, car_orientation = random_rotation(test_offset=args["orientation_offset"], offset=args["orientation_offset_angle"])
    # translation = np.array((20,20))
    translation = random_translation(40)  # if random translation generation should be used
    buggy_perception = 0  # percentage of cones that are thrown out randomly to test robustness
    car_orientation = np.array((1.0, -1.0))

    print("Running Test used following translation: ", translation)
    print("Running Test used following rotation: ", math.degrees(rotation_angle))
    print("Running Test used following rotation: ", car_orientation)

    left_real, right_real = generate_data(rotation_angle, translation,
                                          buggy_perception)  # get testing data for development purposes
    unsorted_data = np.vstack((left_real, right_real))
    print("Running algorithm with only: {} percents of the data points".format(100 * unsorted_data.shape[0] / 58))

    # starting point already defines translation of the skidpad course since the starting point is always the same
    # center_skidpad is used for the beginning straight line and the finishing straight line
    rot_matrix = get_rotational_matrix(rotation_angle)
    # define all the raw points where the relative position is known
    starting_point = np.array((0.0, -15.0))
    center_skidpad = np.array((0.0, 0.0))
    ending_point = np.array((0.0, 30.0))  # car should stop within 25 m after center line so 5 m margin for the planner

    # rotate all these point according to the skidpad angle
    starting_point = np.matmul(rot_matrix, starting_point)
    center_skidpad = np.matmul(rot_matrix, center_skidpad)
    ending_point = np.matmul(rot_matrix, ending_point)

    # translate all these points w.r.t to the car position
    starting_point = np.array((starting_point[0] + translation[0], starting_point[1] + translation[1]))
    center_skidpad = np.array((center_skidpad[0] + translation[0], center_skidpad[1] + translation[1]))
    ending_point = np.array((ending_point[0] + translation[0], ending_point[1] + translation[1]))

    skidpad_planner = SkidpadPlanner(car_orientation, starting_point)
    overall_list = []
    noise_list = [0.0]  # , 0.1,0.2,0.3,0.4,0.5, 0.6, 0.7, 0.8, 0.9, 1]
    num_iterations = 1
    for noise in noise_list:
        # print("Running evaluation for: ", noise)
        radius_list = []
        center_x_list = []
        center_y_list = []
        for k in range(num_iterations):
            start = time.time()
            upper_left, lower_left, upper_right, lower_right, beginning, finish_line, entire_path = skidpad_planner.run_skidpad(
                noise, unsorted_data, car_orientation, starting_point, translation, center_skidpad, ending_point)
            end = time.time()
            print("Skidpad course calculated in {} milliseconds ".format((end - start) * 1000))
            fig = plt.figure(figsize=[10.0, 10.0])  # quadratic resolution
            plt.scatter(upper_right[:,0], upper_right[:,1], color="red")
            plt.scatter(upper_left[:,0], upper_left[:,1], color="blue")
            # plt.scatter(lower_right[:, 0], lower_right[:, 1], color="red")
            # plt.scatter(lower_left[:, 0], lower_left[:, 1], color="blue")
            # plt.scatter(beginning[:,0], beginning[:,1], color="black")
            # plt.scatter(starting_point[0], starting_point[1], label="Start")
            # plt.scatter(finish_line[:,0], finish_line[:,1], color="green")
            plt.plot(entire_path[:, 0], entire_path[:, 1], '.')
            # plt.xlim(starting_point[0] - 30, starting_point[0] + 30)
            # plt.ylim(starting_point[1] - 30, starting_point[1] + 30)
            # plt.scatter(skidpad_planner.cone_positions[:,0], skidpad_planner.cone_positions[:,1])
            # plt.scatter(unsorted_data[:, 0], unsorted_data[:, 1])
            plt.grid()
            plt.show()
