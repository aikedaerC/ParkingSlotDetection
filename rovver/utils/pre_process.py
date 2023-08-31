"""Defines data structure."""
import math
from enum import Enum
import cv2 as cv
import numpy as np


class PointShape(Enum):
    """The point shape types used to pair two marking points into slot."""
    none = 0
    l_down = 1
    t_down = 2
    t_middle = 3
    t_up = 4
    l_up = 5


def direction_diff(direction_a, direction_b):
    """Calculate the angle between two direction."""
    diff = abs(direction_a - direction_b)
    return diff if diff < math.pi else 2*math.pi - diff


def determine_point_shape(point, vector):
    """Determine which category the point is in."""
    bridge_angle_diff = 0.09757113548987695 + 0.1384059287593468
    separator_angle_diff = 0.284967562063968 + 0.1384059287593468

    vec_direct = math.atan2(vector[1], vector[0])
    vec_direct_up = math.atan2(-vector[0], vector[1])
    vec_direct_down = math.atan2(vector[0], -vector[1])
    if point.shape < 0.5:
        if direction_diff(vec_direct, point.direction) < bridge_angle_diff:
            return PointShape.t_middle
        if direction_diff(vec_direct_up, point.direction) < separator_angle_diff:
            return PointShape.t_up
        if direction_diff(vec_direct_down, point.direction) < separator_angle_diff:
            return PointShape.t_down
    else:
        if direction_diff(vec_direct, point.direction) < bridge_angle_diff:
            return PointShape.l_down
        if direction_diff(vec_direct_up, point.direction) < separator_angle_diff:
            return PointShape.l_up
    return PointShape.none


def calc_point_squre_dist(point_a, point_b):
    """Calculate distance between two marking points."""
    distx = point_a[0] - point_b[0]
    disty = point_a[1] - point_b[1]
    return distx ** 2 + disty ** 2


def calc_point_direction_angle(point_a, point_b):
    """Calculate angle between direction in rad."""
    return direction_diff(point_a[2], point_b[2])


def match_marking_points(point_a, point_b):
    """Determine whether a detected point match ground truth."""
    
    squared_distance_thresh = 0.000277778  # 10 pixel in 600*600 image
    direction_angle_thresh = 0.5235987755982988  # 30 degree in rad 
    
    dist_square = calc_point_squre_dist(point_a, point_b)
    #if min(point_a.shape[1], point_b.shape[1]) <= 2:
    if True:
        return dist_square < squared_distance_thresh

    angle = calc_point_direction_angle(point_a, point_b)
    if point_a[3] > 0.5 and point_b[3] < 0.5:
        return False
    if point_a[3] < 0.5 and point_b[3] > 0.5:
        return False
    return (dist_square < squared_distance_thresh
            and angle < direction_angle_thresh)


def match_slots(slot_a, slot_b):
    """Determine whether a detected slot match ground truth."""
    squared_distance_thresh = 0.000277778  # 10 pixel in 600*600 image
    dist_x1 = slot_b[0] - slot_a[0]
    dist_y1 = slot_b[1] - slot_a[1]
    squared_dist1 = dist_x1**2 + dist_y1**2
    dist_x2 = slot_b[2] - slot_a[2]
    dist_y2 = slot_b[3] - slot_a[3]
    squared_dist2 = dist_x2 ** 2 + dist_y2 ** 2
    return (squared_dist1 < squared_distance_thresh
            and squared_dist2 < squared_distance_thresh)



"""Perform data augmentation and preprocessing."""

def boundary_check(centralied_marks):
    """Check situation that marking point appears too near to border."""
    for mark in centralied_marks:
        if mark[0] < -260 or mark[0] > 260 or mark[1] < -260 or mark[1] > 260:
            return False
    return True

def overlap_check(centralied_marks):
    """Check situation that multiple marking points appear in same cell."""
    for i in range(len(centralied_marks) - 1):
        i_x = centralied_marks[i, 0]
        i_y = centralied_marks[i, 1]
        for j in range(i + 1, len(centralied_marks)):
            j_x = centralied_marks[j, 0]
            j_y = centralied_marks[j, 1]
            if abs(j_x - i_x) < 600 / 16 and abs(j_y - i_y) < 600 / 16:
                return False
    return True

def generalize_marks(centralied_marks, with_direction=False):
    """Convert coordinate to [0, 1] and calculate direction label."""
    generalized_marks = []
    for mark in centralied_marks:
        xval = (mark[0] + 300) / 600
        yval = (mark[1] + 300) / 600
        if with_direction:
            direction = math.atan2(mark[3] - mark[1], mark[2] - mark[0])
            generalized_marks.append([xval, yval, direction, mark[4]])
        else:
            generalized_marks.append([xval, yval])
    return np.array(generalized_marks)

def rotate_vector(vector, angle_degree):
    """Rotate a vector with given angle in degree."""
    angle_rad = math.pi * angle_degree / 180
    xval = vector[0]*math.cos(angle_rad) + vector[1]*math.sin(angle_rad)
    yval = -vector[0]*math.sin(angle_rad) + vector[1]*math.cos(angle_rad)
    return xval, yval

def rotate_centralized_marks(centralied_marks, angle_degree):
    """Rotate centralized marks with given angle in degree."""
    rotated_marks = centralied_marks.copy()
    for i in range(centralied_marks.shape[0]):
        mark = centralied_marks[i]
        rotated_marks[i, 0:2] = rotate_vector(mark[0:2], angle_degree)
        rotated_marks[i, 2:4] = rotate_vector(mark[2:4], angle_degree)
    return rotated_marks

def rotate_image(image, angle_degree):
    """Rotate image with given angle in degree."""
    rows, cols, _ = image.shape
    rotation_matrix = cv.getRotationMatrix2D((rows/2, cols/2), angle_degree, 1)
    return cv.warpAffine(image, rotation_matrix, (rows, cols))
