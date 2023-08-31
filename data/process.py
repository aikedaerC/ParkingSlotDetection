"""Defines related function to process defined data structure."""
import math
import numpy as np
import torch
import config
from data.struct import MarkingPoint, detemine_point_shape, PredMarkingPoint


def non_maximum_suppression(pred_points):
    """Perform non-maxmum suppression on marking points."""
    suppressed = [False] * len(pred_points)
    for i in range(len(pred_points) - 1):
        for j in range(i + 1, len(pred_points)):
            i_x = pred_points[i][1].x
            i_y = pred_points[i][1].y
            j_x = pred_points[j][1].x
            j_y = pred_points[j][1].y
            # 0.0625 = 1 / 16
            if abs(j_x - i_x) < 0.0625 and abs(j_y - i_y) < 0.0625:
                idx = i if pred_points[i][0] < pred_points[j][0] else j
                suppressed[idx] = True # drop those True, becasuse the pair is so close
    if any(suppressed):
        unsupres_pred_points = []
        for i, supres in enumerate(suppressed):
            if not supres:
                unsupres_pred_points.append(pred_points[i])
        return unsupres_pred_points
    return pred_points


def get_predicted_points(prediction, thresh):
    """Get marking points from one predicted feature map.  """
    assert isinstance(prediction, torch.Tensor)
    predicted_points = []
    prediction = prediction.detach().cpu().numpy() # [shape, confidence, x, y, cos(direction), sin(direction), x_third, y_third, confidence_third]
    for i in range(prediction.shape[1]):
        for j in range(prediction.shape[2]):
            if prediction[1, i, j] >= thresh and prediction[-1,i,j] >= thresh:
                xval = (j + prediction[2, i, j]) / prediction.shape[2]
                yval = (i + prediction[3, i, j]) / prediction.shape[1] 
                if not (config.BOUNDARY_THRESH <= xval <= 1-config.BOUNDARY_THRESH
                        and config.BOUNDARY_THRESH <= yval <= 1-config.BOUNDARY_THRESH):
                    continue
                cos_value = prediction[4, i, j]
                sin_value = prediction[5, i, j]
                direction = math.atan2(sin_value, cos_value)
                
                xval_third = (j + prediction[6, i, j]) / prediction.shape[2]
                yval_third = (i + prediction[7, i, j]) / prediction.shape[1]
                if not (config.BOUNDARY_THRESH <= xval_third <= 1-config.BOUNDARY_THRESH
                        and config.BOUNDARY_THRESH <= yval_third <= 1-config.BOUNDARY_THRESH):
                    continue
                marking_point = PredMarkingPoint(
                    xval, yval, direction, prediction[0, i, j], xval_third, yval_third) # ['x', 'y', 'direction', 'shape', 'x_third', 'y_third']
                predicted_points.append((prediction[1, i, j], marking_point))
    # print(f"pred points: {predicted_points}")
    # import pdb; pdb.set_trace()
    return non_maximum_suppression(predicted_points)


def pass_through_third_point(marking_points, i, j): 
    """See whether the line between two points pass through a third point."""
    # list of ['x', 'y', 'direction', 'shape', 'x_third', 'y_third']
    x_1 = marking_points[i].x
    y_1 = marking_points[i].y
    x_2 = marking_points[j].x
    y_2 = marking_points[j].y
    for point_idx, point in enumerate(marking_points):
        if point_idx == i or point_idx == j:
            continue
        # see if the third point between (x1,y1) and (x2,y2)
        x_0 = point.x
        y_0 = point.y
        vec1 = np.array([x_0 - x_1, y_0 - y_1])
        vec2 = np.array([x_2 - x_0, y_2 - y_0])
        vec1 = vec1 / np.linalg.norm(vec1)
        vec2 = vec2 / np.linalg.norm(vec2)
        if np.dot(vec1, vec2) > config.SLOT_SUPPRESSION_DOT_PRODUCT_THRESH:
            return True
    return False


def pair_marking_points(point_a, point_b, img_name):
    """See whether two marking points form a slot.
        none = 0
        l_down = 1
        t_down = 2
        t_middle = 3
        t_up = 4
        l_up = 5
    """

    vector_ab = np.array([point_b.x - point_a.x, point_b.y - point_a.y])
    vector_ab = vector_ab / np.linalg.norm(vector_ab)
    point_shape_a = detemine_point_shape(point_a, vector_ab, img_name)
    point_shape_b = detemine_point_shape(point_b, -vector_ab, img_name)
    if point_shape_a.value == 0 or point_shape_b.value == 0:
        return 0
    if point_shape_a.value == 3 and point_shape_b.value == 3:
        return 0
    if point_shape_a.value > 3 and point_shape_b.value > 3:
        return 0
    if point_shape_a.value < 3 and point_shape_b.value < 3:
        return 0
    if point_shape_a.value != 3:
        if point_shape_a.value > 3:
            return 1
        if point_shape_a.value < 3:
            return -1
    if point_shape_a.value == 3:
        if point_shape_b.value < 3:
            return 1
        if point_shape_b.value > 3:
            return -1
