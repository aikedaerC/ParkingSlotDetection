import os
import torch
import cv2
import numpy as np
import json
import math
from tqdm import tqdm
import random
from torchvision.transforms import ToTensor

import config
from model import DirectionalPointDetector
from data import get_predicted_points, pair_marking_points, calc_point_squre_dist, pass_through_third_point

mode = 'online' # online
if mode == 'online':
    display = False
    img_dir = '/work/data/visual-parking-space-line-recognition-test-set/'
    json_dir = '/work/output/'
    weights_dir = 'weights/model.pth'
elif mode == 'local':
    display = True
    img_dir = r'F:\ubunut_desktop_back\ParkingSlotDetection\data_comp\train\imgs'
    json_dir = r'F:\ubunut_desktop_back\ParkingSlotDetection\output'
    weights_dir = r'F:\ubunut_desktop_back\ParkingSlotDetection\DMPR-PS-master\weights\dmpr_pretrained_weights.pth'
    os.makedirs(json_dir, exist_ok=True)



def inference_slots(marking_points):
    """Inference slots based on marking points."""
    num_detected = len(marking_points)
    slots = []
    for i in range(num_detected - 1):
        for j in range(i + 1, num_detected):
            point_i = marking_points[i]
            point_j = marking_points[j]
            # Step 1: length filtration.
            distance = calc_point_squre_dist(point_i, point_j)
            if not (config.VSLOT_MIN_DIST <= distance <= config.VSLOT_MAX_DIST
                    or config.HSLOT_MIN_DIST <= distance <= config.HSLOT_MAX_DIST):
                continue
            # Step 2: pass through filtration.
            if pass_through_third_point(marking_points, i, j):
                continue
            result = pair_marking_points(point_i, point_j)
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


def dot(p1, p2):
    return p1[0] * p2[0] + p1[1] * p2[1]


def get_angle(x1, y1, x2, y2):
    vec_bc = (x2 - x1, y2 - y1)
    horizon = (-1, 0)
    costheta = dot(vec_bc, horizon) / (vec_bc[0] ** 2 + vec_bc[1] ** 2) ** 0.5
    theta = np.arccos(costheta)
    if vec_bc[1] > 0:
        theta = -theta
    return theta
    

def detect_image(detector, device):
    from compare_nonslot import slash_rectangle, rectangle
    ill = list(rectangle)
    # ill = os.listdir(img_dir)
    # ill = random.sample(ill, k=77) + ill_sp
    # random.shuffle(ill)
    # ill = os.listdir(img_dir)
    none_det = []
    shape_list = []
    for idx, img_name in enumerate(tqdm(ill)):
        img_path = os.path.join(img_dir, img_name)
        image = cv2.imread(img_path)
        pred_points = detect_marking_points(detector, image, 0.5, device)
        marking_points = [e[1] for e in pred_points]
        # for pp in marking_points:
        #     shape_list.append(pp.shape)
        slots = inference_slots(marking_points)
        height = image.shape[0]
        width = image.shape[1]
        
        slots_list = []
        junctions = []
        for slot in slots:
            point_a = marking_points[slot[0]]
            point_b = marking_points[slot[1]]
            p0_x = width * point_a.x - 0.5
            p0_y = height * point_a.y - 0.5
            p1_x = width * point_b.x - 0.5
            p1_y = height * point_b.y - 0.5
            x1, y1, x2, y2 = p0_x, p0_y, p1_x, p1_y
            vec = np.array([p1_x - p0_x, p1_y - p0_y])
            vec = vec / np.linalg.norm(vec)
            distance = calc_point_squre_dist(point_a, point_b)
            if config.VSLOT_MIN_DIST <= distance <= config.VSLOT_MAX_DIST:
                separating_length = config.LONG_SEPARATOR_LENGTH
            elif config.HSLOT_MIN_DIST <= distance <= config.HSLOT_MAX_DIST:
                separating_length = config.SHORT_SEPARATOR_LENGTH
            p2_x = p0_x + height * separating_length * vec[1]
            p2_y = p0_y - width * separating_length * vec[0]
            p3_x = p1_x + height * separating_length * vec[1]
            p3_y = p1_y - width * separating_length * vec[0]

            if display:
                p0_x = int(round(p0_x))
                p0_y = int(round(p0_y))
                p1_x = int(round(p1_x))
                p1_y = int(round(p1_y))
                p2_x = int(round(p2_x))
                p2_y = int(round(p2_y))
                p3_x = int(round(p3_x))
                p3_y = int(round(p3_y))
                cv2.line(image, (p0_x, p0_y), (p1_x, p1_y), (255, 0, 0), 2)
                cv2.line(image, (p0_x, p0_y), (p2_x, p2_y), (255, 0, 0), 2)
                cv2.line(image, (p1_x, p1_y), (p3_x, p3_y), (255, 0, 0), 2)
                junctions.append((p0_x, p0_y))
                junctions.append((p1_x, p1_y))

            angle1 = get_angle(p0_x, p0_y, p2_x, p2_y)
            angle2 = get_angle(p1_x, p1_y, p3_x, p3_y)
            slot_dict = {
                "points": [[x1, y1], [x2, y2]],
                "angle1": angle1,
                "angle2": angle2,
                "scores": 0.95,
            }
            slots_list.append(slot_dict)

        slots_dict = {"slot": slots_list}
        if len(slots_list) == 0:
            none_det.append(img_name)
            print(f"{img_name} detect nothing!")
        json_name = img_name.replace('.jpg', '.json')
        json_path = os.path.join(json_dir, json_name)
        with open(json_path, 'w') as f:
            json.dump(slots_dict, f)

        if display:
            for junction in junctions:
                cv2.circle(image, junction, 3, (0, 0, 255), 4)
            cv2.imshow('demo', image)
            cv2.waitKey(200)

    print(f"shape_list: {shape_list}")
    print(f"nothing detected in these img: {none_det}, in total: {len(none_det)}")

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