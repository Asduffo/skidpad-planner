# script defines some helper functions useful for skidpad testing
import numpy as np
import math
import matplotlib.pyplot as plt

def get_rotational_matrix(skidpad_angle):
    """

    :param skidpad_angle: skidpad angle in radians
    :return: rotational matrix as 2,2 np-array
    """
    rotational_matrix = np.zeros((2, 2))
    rotational_matrix[0, 0] = math.cos(skidpad_angle)
    rotational_matrix[0, 1] = (-1) * math.sin(skidpad_angle)
    rotational_matrix[1, 0] = math.sin(skidpad_angle)
    rotational_matrix[1, 1] = math.cos(skidpad_angle)

    return rotational_matrix

def random_translation(max):
    '''
    generates random values for the translation for testing purposes
    :param max: max-value for both x- and y-axis
    :return: randomly generated translation within the given intervall
    '''
    direc_x = np.random.random_integers(2)
    direc_y = np.random.random_integers(2)
    x_trans = np.random.random_integers(max)
    y_trans = np.random.random_integers(max)

    if direc_x==1:
        x_trans *= (-1)
    if direc_y==1:
        y_trans *= (-1)

    return np.array((x_trans, y_trans))

def random_rotation(test_offset, offset):
    '''
    computes randomly a skidpad rotation
    :param test_offset
    :return: rotation and corresponding car orientation
    '''
    rot_sign = np.random.random_integers(2)
    x_value = 1.0
    rotation = np.random.uniform(0.0, math.pi/2)

    if not test_offset:
        y_value = x_value / (math.tan(rotation))

    else:
        y_value = x_value / (math.tan(rotation + math.radians(offset)))



    # choose randomly one quadrant where the vector is mapped to
    rot_quadrant = np.random.random_integers(4)
    if rot_quadrant == 1:
        return (-1) * rotation, np.array((x_value, y_value))
    if rot_quadrant == 2:
        return rotation, np.array(((-1) * x_value, y_value))
    if rot_quadrant == 3:
        return abs(math.pi - rotation), np.array(((-1) * x_value, (-1) * y_value))
    if rot_quadrant == 4:
        return (-1) * abs(math.pi - rotation), np.array((x_value, (-1) * y_value))




def preprocess_data(x, y):
    result = []
    for first, second in zip(x, y):
        result.append([first, second])
    return np.array(result)

def rotate_cones(rotational_matrix, cones):
    right_half_new = []
    for element in cones:
        new_element = np.matmul(rotational_matrix, element)
        right_half_new.append(new_element)
    return np.array(right_half_new)

def kill_cones(amount, size):
    '''

    :param amount: amount of cones that should be sorted out
    :return: cone mask that randomly rejects cones
    '''
    cone_mask = np.random.choice([0, 1], size=(size,), p=[amount,1-amount])
    cone_mask = np.array(cone_mask, dtype=bool)
    return cone_mask

def translate_cones(cone_list, translation):
    '''
    translates two n,2 arrays of cones in a given direction
    :param left_cones:
    :param right_cones:
    :param translation:
    :return: two n,2 np.arrays for left and right translated cones
    '''
    first_real_new = []
    for cone_left in cone_list:
        cone_left = np.array((cone_left[0] + translation[0], cone_left[1] + translation[1]))
        first_real_new.append(cone_left)

    return np.array(first_real_new)

def generate_data(rotation, translation, buggy_perception):

    '''
    This function generates data points that simulate a skidpad course with a given rotation and translation
    :param rotation: rotation given in radians
    :param translation: given as np.array [x,y]
    :param buggy_perception: simulates a suboptimal perception
    :return: n,2 np-array of points
    '''

    x = [7.625, 7.044, 5.391, 2.918, 0, -2.918, -5.391, -7.044, -7.625, -7.044, -5.391, -2.918, 0, 2.918, 5.391,
                  7.044, 7.513, 4.066, 0, -4.066, -7.513, -9.816, -10.625, -9.816, -7.513, -4.066, 0, 4.066, 7.513]
    y = [0.0, 2.918, 5.391, 7.044, 7.625, 7.044, 5.391, 2.918, 0, -2.918, -5.391, -7.044, -7.625, -7.044, -5.391,
                  -2.918, 7.513, 9.816, 10.625, 9.816, 7.513, 4.066, 0, -4.066, -7.513, -9.816, -10.625, -9.816, -7.513]

    x = [r-9.125 for r in x]
    z = [xr*(-1) for xr in x]

    right_half = preprocess_data(x,y)
    left_half = preprocess_data(z,y)


    rot_angle = rotation
    rotational_matrix = get_rotational_matrix(rot_angle)

    cone_mask_left = kill_cones(buggy_perception, left_half.shape[0])
    cone_mask_right = kill_cones(buggy_perception, right_half.shape[0])


    right_half_new = rotate_cones(rotational_matrix, right_half)
    left_half_new = rotate_cones(rotational_matrix, left_half)

    left_half_new= translate_cones(left_half_new, translation)
    right_half_new = translate_cones(right_half_new, translation)

    # simulate buggy perception if wanted that randomly kills cones
    # if not set amount to 0
    left_half_new = left_half_new[cone_mask_left,:]
    right_half_new = right_half_new[cone_mask_right,:]

    return left_half_new, right_half_new


#fig = plt.figure(figsize=[6.4,6.4])
#plt.scatter(right_half[:,0], right_half[:,1])
#plt.scatter(right_half_new[:,0], right_half_new[:,1], color="red")
#plt.scatter(left_half[:,0],left_half[:,1])
#plt.scatter(left_half_new[:,0],left_half_new[:,1])
#plt.grid()
#plt.show()