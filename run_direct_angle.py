import os
import torch
import cv2
import numpy as np
import json
import math
from tqdm import tqdm
from torchvision.transforms import ToTensor

import config
from model import DirectionalPointDetector
from data import get_predicted_points, pair_marking_points, calc_point_squre_dist, pass_through_third_point, MarkingPoint

mode = 'local' # online
if mode == 'online':
    display = False
    img_dir = '/work/data/visual-parking-space-line-recognition-test-set/'
    json_dir = '/work/output/'
    weights_dir = 'weights/model.pth'
elif mode == 'local':
    display = True
    # img_dir = r'F:\ubunut_desktop_back\ParkingSlotDetection\data_comp\train\imgs'
    img_dir = r'F:\ubunut_desktop_back\ParkingSlotDetection\data_extra\train\aug\train'
    json_dir = r'F:\ubunut_desktop_back\ParkingSlotDetection\output'
    weights_dir = r'F:\ubunut_desktop_back\ParkingSlotDetection\NEW_DMPRbakup\weights\dp_detector_2.pth'
    os.makedirs(json_dir, exist_ok=True)



def inference_slots(marking_points, img_name):
    """Inference slots based on marking points."""
    num_detected = len(marking_points)
    slots = []
    for i in range(num_detected - 1):
        for j in range(i + 1, num_detected): # O(N^2) searching
            point_i = marking_points[i] # one marking point ['x', 'y', 'direction', 'shape', 'x_third', 'y_third']
            point_j = marking_points[j] # one marking point ['x', 'y', 'direction', 'shape', 'x_third', 'y_third']
            # Step 1: length filtration.
            distance = calc_point_squre_dist(point_i, point_j)
            if not (config.VSLOT_MIN_DIST <= distance <= config.VSLOT_MAX_DIST
                    or config.HSLOT_MIN_DIST <= distance <= config.HSLOT_MAX_DIST):
                print(f"img {img_name}, marking_point_{i} and {j} distance not passed!")
                continue
            # Step 2: pass through filtration.
            if pass_through_third_point(marking_points, i, j):  # list of ['x', 'y', 'direction', 'shape', 'x_third', 'y_third']
                print(f"img {img_name}, marking_point_{i} and {j} third point not passed!")
                continue
            result = pair_marking_points(point_i, point_j, img_name)
            if result == 1:
                slots.append((i, j))
            elif result == -1:
                slots.append((j, i))
    return slots


def preprocess_image(image):
    """Preprocess numpy image to torch tensor."""
    if image.shape[0] != 512 or image.shape[1] != 512:
        image = cv2.resize(image, (512, 512))
    return torch.unsqueeze(ToTensor()(image), 0)


def detect_marking_points(detector, image, thresh, device):
    """Given image read from opencv, return detected marking points."""
    prediction = detector(preprocess_image(image).to(device))
    return get_predicted_points(prediction[0], thresh)


def calculate_angle(vector_a, vector_b):
    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))  # 计算点积
    norm_a = math.sqrt(sum(a * a for a in vector_a))              # 计算向量a的模
    norm_b = math.sqrt(sum(b * b for b in vector_b))              # 计算向量b的模
    cos_theta = dot_product / (norm_a * norm_b)                   # 计算两个向量的夹角的余弦值
    angle_radians = math.acos(max(-1.0, min(1.0, cos_theta)))  # 计算夹角的弧度值
    return angle_radians


def detect_image(detector, device):
    ill = os.listdir(img_dir)
    ill = [e for e in ill if e.endswith(".jpg")]
    none_det = []
    for idx, img_name in enumerate(tqdm(ill)):
        img_path = os.path.join(img_dir, img_name)
        # detected
        image = cv2.imread(img_path)
        # import pdb; pdb.set_trace()

        pred_points = detect_marking_points(detector, image, 0.8, device)
        marking_points = [e[1] for e in pred_points]

        height = image.shape[0]
        width = image.shape[1]
        print(f"{len(marking_points)} marking_points has been detected!")
        for mpoint in marking_points:
            cv2.circle(image, (int(round(width * mpoint.x - 0.5)), int(round(width * mpoint.y - 0.5))), 3, (0, 0, 255), 4)
            cv2.circle(image, (int(round(width * mpoint.x_third - 0.5)), int(round(width * mpoint.y_third - 0.5))), 3, (0, 255,0), 4)
        cv2.imshow('demo', image)
        cv2.waitKey(1000)

        # label
        # img_label = img_path.replace(".jpg", ".json")
        # marking_points_lab = []
        # with open(os.path.join(img_label), 'r') as file:
        #     for label in json.load(file):
        #         marking_points_lab.append(MarkingPoint(*label))

        # for mp in marking_points_lab:
        #     p0_x = 512 * mp.x - 0.5
        #     p0_y = 512 * mp.y - 0.5
        #     p2_x = p0_x - 10 * math.cos(mp.angle)
        #     p2_y = p0_y - 10 * math.sin(mp.angle)
        #     cv2.circle(image, (int(round(p0_x)), int(round(p0_y))), 3, (0, 0, 255), 4)
        #     cv2.circle(image, (int(round(p2_x)), int(round(p2_y))), 3, (0, 255, 0), 4)
        # cv2.imshow('demo', image)
        # cv2.waitKey(500)


    #     slots = inference_slots(marking_points, img_name)
    #     slots_list = []
    #     junctions = []
    #     for slot in slots:
    #         point_a = marking_points[slot[0]]
    #         point_b = marking_points[slot[1]]
    #         p0_x = width * point_a.x - 0.5
    #         p0_y = height * point_a.y - 0.5
    #         p1_x = width * point_b.x - 0.5
    #         p1_y = height * point_b.y - 0.5

    #         p0_x_third = width * point_a.x_third - 0.5
    #         p0_y_third = height * point_a.y_third - 0.5
    #         p1_x_third = width * point_b.x_third - 0.5
    #         p1_y_third = height * point_b.y_third - 0.5
            
    #         distance = calc_point_squre_dist(point_a, point_b)
    #         if config.VSLOT_MIN_DIST <= distance <= config.VSLOT_MAX_DIST:
    #             separating_length = config.LONG_SEPARATOR_LENGTH
    #         elif config.HSLOT_MIN_DIST <= distance <= config.HSLOT_MAX_DIST:
    #             separating_length = config.SHORT_SEPARATOR_LENGTH
    #         # print(f"angle: {point_a.angle/np.pi * 180}")
    #         # 大多数是90度的情况，那么就令离垂直差距大的情况，为预测的third_point和point_a的向量的角度为预测值
    #         # result = pair_marking_points(point_a, point_b)  # 0 for 平行四边形,暂时先不用
    #         vec_seprator = (point_a.x_third - point_a.x, point_a.y_third - point_a.y)
    #         # vec_entrance = (point_a.x - point_b.x, point_a.y - point_b.y)
    #         angle = calculate_angle(vec_seprator, (-1, 0))
    #         print(f"angle: {angle}")
            
    #         # 这下面没问题，关键在角度, 角度因该满足在（-1，0）上为正，在（-1，0）下为负，并且绝对值都在【0,pi】
    #         # p2_x = p0_x - 100 * math.cos(angle)
    #         # p2_y = p0_y - 100 * math.sin(angle)
    #         # p3_x = p1_x - 100 * math.cos(angle)
    #         # p3_y = p1_y - 100 * math.sin(angle)

    #         if display:
    #             cv2.line(image, (int(p0_x), int(p0_y)), (int(p1_x), int(p1_y)), (0, 255, 0), 2)
    #             cv2.line(image, (int(p0_x), int(p0_y)), (int(p0_x_third), int(p0_y_third)), (255, 0, 0), 2)
    #             cv2.line(image, (int(p1_x), int(p1_y)), (int(p1_x_third), int(p1_y_third)), (255, 0, 0), 2)
    #             junctions.append((int(p0_x), int(p0_y)))
    #             junctions.append((int(p1_x), int(p1_y)))
            
    #         slot_dict = {
    #             "points": [[p0_x, p0_y], [p1_x, p1_y]],
    #             "angle1": float(angle),
    #             "angle2": float(angle),
    #             "scores": 0.95,
    #         }
    #         slots_list.append(slot_dict)

    #     slots_dict = {"slot": slots_list}
    #     if len(slots_list) == 0:
    #         none_det.append(img_name)
    #         print(f"{img_name} detect nothing!")
    #     json_name = img_name.replace('.jpg', '.json')
    #     json_path = os.path.join(json_dir, json_name)
    #     with open(json_path, 'w') as f:
    #         json.dump(slots_dict, f)

    #     if display:
    #         for junction in junctions:
    #             cv2.circle(image, junction, 3, (0, 0, 255), 4)
    #         cv2.imshow('demo', image)
    #         cv2.waitKey(5000)

    # print(f"nothing detected in these img: {none_det}, in total: {len(none_det)}")

def inference_detector():
    device = torch.device('cuda:0')
    torch.set_grad_enabled(False)
    dp_detector = DirectionalPointDetector(3, 32, config.NUM_FEATURE_MAP_CHANNEL)
    dp_detector.load_state_dict(torch.load(weights_dir, map_location=device))
    dp_detector.to(device)
    dp_detector.eval()
    detect_image(dp_detector, device)


if __name__ == '__main__':
    inference_detector()