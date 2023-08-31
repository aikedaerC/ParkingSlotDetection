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
    weights_dir = r'F:\ubunut_desktop_back\ParkingSlotDetection\DMPR-PS-master\weights\iter4217_loss0.0015.pth'
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
    # from compare_nonslot import dmpr
    ill = os.listdir(img_dir)
    # later_dmpr56 = ['image20160722192751_3732.jpg', '20161019-2-653.jpg', 'p2_img27_3312.jpg', 'image20160722192751_1776.jpg', 'img9_1290.jpg', 'img5_7005.jpg', 'p2_img23_0648.jpg', 'p2_img118_0420.jpg', 'image20160722192751_3720.jpg', 'image20160722192751_3316.jpg', 'img5_7032.jpg', 'p2_img118_0042.jpg', 'img5_6957.jpg', 'image20160722192751_3320.jpg', 'img5_7521.jpg', 'p2_img12_0234.jpg', 'p2_img118_0078.jpg', 'image20160722192751_3612.jpg', 'image20160722192751_3740.jpg', 'image20160722192751_3736.jpg', 'img9_1254.jpg', 'img5_6963.jpg', 'image20160725143627_560.jpg', 'image20160722192751_4752.jpg', 'p2_img104_0150.jpg', 'image20160722192751_4500.jpg', 'img5_6999.jpg', 'image20160725152215_252.jpg', 'p2_img117_0516.jpg', 'img9_4872.jpg', 'p2_img25_0552.jpg', 'p2_img15_0534.jpg', 'img5_7011.jpg', 'img5_6972.jpg', 'img9_6435.jpg', 'p2_img26_0372.jpg', 'p2_img34_0546.jpg', 'p2_img23_0600.jpg', 'image20160725151308_308.jpg', 'img9_1158.jpg', 'img5_6867.jpg', 'image20160722192751_2376.jpg', 'img8_1740.jpg', 'image20160725142318_1156.jpg', 'image20160722192751_3728.jpg', 'p2_img23_0582.jpg', 'p2_img86_3972.jpg', 'image20160722193621_860.jpg', 'image20160725142318_1152.jpg', 'p2_img25_0528.jpg', 'p2_img25_0516.jpg', 'p2_img13_0444.jpg', 'img5_7023.jpg', 'p2_img23_0654.jpg', 'image20160725142318_1116.jpg', 'image20160722192751_1780.jpg']
    # dmpr = set(('image20160722192751_3720.jpg', 'p2_img25_0528.jpg', 'img7_1557.jpg', 'image20160722192751_3612.jpg', 'image20160722192751_3604.jpg', 'img8_1740.jpg', 'img5_6867.jpg', 'img5_6213.jpg', 'p2_img10_0126.jpg', 'p2_img49_3156.jpg', 'image20160722192751_3748.jpg', 'image20160722192751_3316.jpg', 'p2_img29_1788.jpg', 'p2_img83_3870.jpg', 'img5_7023.jpg', 'p2_img53_0054.jpg', 'image20160722193621_1316.jpg', 'p2_img104_0150.jpg', 'img5_7521.jpg', 'p2_img11_0468.jpg', 'p2_img51_3618.jpg', 'p2_img34_1938.jpg', 'image20160722192751_3728.jpg', 'image20160722192751_2960.jpg', 'p2_img118_0312.jpg', 'p2_img21_2292.jpg', 'image20160722192751_3192.jpg', 'p2_img27_3312.jpg', 'image20160722192751_3248.jpg', '20161111-03-362.jpg', 'img9_4872.jpg', 'p2_img8_0294.jpg', 'img9_1290.jpg', 'image20160722192751_2376.jpg', 'img5_7032.jpg', 'p2_img118_0288.jpg', 'image20160722192751_1076.jpg', 'p2_img26_0384.jpg', 'image20160725142318_1116.jpg', 'img9_6435.jpg', 'image20160725152215_252.jpg', 'img5_9111.jpg', 'img5_6990.jpg', '20161019-2-653.jpg', 'image20160722192751_1776.jpg', 'p2_img116_0114.jpg', 'image20160725142318_1156.jpg', 'p2_img21_2214.jpg', 'p2_img15_0732.jpg', 'p2_img118_0420.jpg', 'p2_img119_0348.jpg', 'image20160722192751_3732.jpg', 'image20160722192751_3736.jpg', 'image20160722192751_3608.jpg', 'image20160725142318_1112.jpg', 'image20160722192751_3752.jpg', 'p2_img21_1452.jpg', 'image20160722192751_3488.jpg', 'img5_6996.jpg', 'p2_img15_1686.jpg', 'p2_img25_0552.jpg', 'img5_6957.jpg', 'p2_img19_0852.jpg', 'p2_img118_0480.jpg', 'img5_6963.jpg', 'image20160722192751_3480.jpg', 'image20160725151308_308.jpg', 'p2_img52_1878.jpg', 'image20160722192751_2920.jpg', 'p2_img34_0546.jpg', 'img5_6987.jpg', 'p2_img31_0090.jpg', 'p2_img28_4140.jpg', 'p2_img119_0306.jpg', 'p2_img23_0582.jpg', 'image20160722192751_3496.jpg', 'p2_img28_0330.jpg', 'p2_img12_0234.jpg', 'image20160722192751_3600.jpg', 'img5_6999.jpg', 'p2_img86_3972.jpg', 'img9_1158.jpg', 'p2_img52_2004.jpg', 'p2_img25_0516.jpg', 'p2_img118_0078.jpg', 'img5_6900.jpg', 'img5_7005.jpg', 'p2_img12_0702.jpg', 'img5_6972.jpg', 'image20160722192751_4752.jpg', 'image20160725143627_560.jpg', 'p2_img26_0372.jpg', 'image20160722192751_1780.jpg', 'p2_img23_0648.jpg', 'image20160722192751_1740.jpg', 'p2_img117_0516.jpg', 'p2_img27_3330.jpg', 'image20160722192751_3208.jpg', 'p2_img18_0294.jpg', 'image20160722192751_4500.jpg', 'image20160725142318_1152.jpg', 'image20160722193621_860.jpg', 'p2_img23_0654.jpg', 'p2_img23_1638.jpg', 'p2_img118_0042.jpg', 'p2_img15_0552.jpg', 'image20160722192751_3320.jpg', 'p2_img26_2748.jpg', 'p2_img10_0114.jpg', 'image20160722192751_3484.jpg', 'p2_img117_0594.jpg', 'p2_img13_0444.jpg', 'p2_img118_0258.jpg', 'p2_img21_2712.jpg', 'image20160722192751_3740.jpg', 'img7_0387.jpg', 'p2_img15_0534.jpg', 'img5_7011.jpg', 'p2_img25_0510.jpg', 'p2_img23_0600.jpg', 'img9_1254.jpg', 'image20160722193621_972.jpg', 'img3_5574.jpg'))
    # ill = list(dmpr)

    none_det = []
    for idx, img_name in enumerate(tqdm(ill)):
        img_path = os.path.join(img_dir, img_name)
        image = cv2.imread(img_path)
        pred_points = detect_marking_points(detector, image, 0.5, device)
        marking_points = [e[1] for e in pred_points]
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
                "angle1": 124.9,
                "angle2": 124.9,
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
            cv2.waitKey(500)

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