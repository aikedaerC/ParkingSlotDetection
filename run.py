import cv2
import time
import torch
import pprint
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
from rovver.utils.config import get_logger, cfg_from_file, merge_new_config
from permissive_dict import PermissiveDict as Dict
from rovver.models.builder import build_model
import os
import json

def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument( '-c', '--cfg',      metavar='C',           default="rovver/yamls/ps_gat.yaml",                  help='The Configuration file')
    argparser.add_argument( '-s', '--seed',     default=100,           type=int,                                                    help='The random seed')
    argparser.add_argument( '-m', '--ckpt',     type=str,              help='The model path, /psv/v0/cache/check_pretrained/checkpoint_epoch_200.pth')
    argparser.add_argument( '--local_rank',     type=int,              default=0,                                                   help='local rank for distributed training')
    argparser.add_argument( '--launcher',       type=str,              default='none',                                              help='launcher for distributed training')
    argparser.add_argument('--eval_all',        action='store_true',   default=False,                                               help='whether to evaluate all checkpoints')
    args = argparser.parse_args()
    return args

def get_config():
    args = get_args()
    config_file = args.cfg
    random_seed = args.seed
    config = cfg_from_file(config_file)
    config.eval_all = args.eval_all
    config.local_rank = args.local_rank
    config.ckpt = args.ckpt
    config.launcher = args.launcher
    config.random_seed = random_seed
    cfg = Dict()
    merge_new_config(config=cfg, new_config=config)
    return cfg

def main():

    mode = 'local' # online
    if mode == 'online':
        display = False
        img_dir = '/work/data/visual-parking-space-line-recognition-test-set/'
        json_dir = '/work/output/'
        weights_dir = 'model/model.pth'
    elif mode == 'local':
        display = True
        img_dir = r'F:\ubunut_desktop_back\ParkingSlotDetection\data_comp\train\imgs'
        json_dir = r'F:\ubunut_desktop_back\ParkingSlotDetection\output'
        weights_dir = 'model/model.pth'
        withlabel_img_path = r'F:\ubunut_desktop_back\ParkingSlotDetection\output'
        os.makedirs(withlabel_img_path, exist_ok=True)
        os.makedirs(json_dir, exist_ok=True)

    cfg = get_config()
    model = build_model(cfg.model)
    model.load_params_from_file(filename=weights_dir, to_cpu=False)
    model.cuda()
    model.eval()
    
    with torch.no_grad():
        from compare_nonslot import dmpr, slash_rectangle
        img_list = list(slash_rectangle)
        # img_list = os.listdir(img_dir)
        none_dect = []
        for idx, img_name in enumerate(tqdm(img_list)):
            img_path = os.path.join(img_dir, img_name)
            img = cv2.imread(img_path)
            img0 = cv2.resize(img, (512, 512))
            img = img0 / 255.

            data_dict = {}
            data_dict['image'] = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0).cuda()
            pred_dicts, ret_dict = model(data_dict)  # points_angle_pred_batch, slots_pred
            if display:
                car = cv2.imread('rovver/utils/car/car.png')
                car = cv2.resize(car, (512, 512))
                img0[145:365, 210:300] = 0
                img0 += car
                draw_parking_slot(img0, img_name, json_dir, pred_dicts, display, none_dect, withlabel_img_path)
            else:
                draw_parking_slot(img0, img_name, json_dir, pred_dicts, display, none_dect)
        print(f"total none_dect : {len(none_dect)} \n{none_dect}")

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
    

def draw_parking_slot(image, img_name,json_dir, pred_dicts,display, none_dect, output_dir=None):
    slots_pred = pred_dicts['slots_pred']
    width = 512
    height = 512
    VSLOT_MIN_DIST = 0.044771278151623496
    VSLOT_MAX_DIST = 0.1099427457599304
    HSLOT_MIN_DIST = 0.15057789144568634
    HSLOT_MAX_DIST = 0.44449496544202816

    SHORT_SEPARATOR_LENGTH = 0.199519231
    LONG_SEPARATOR_LENGTH = 0.46875
    slots_list = []
    junctions = []
    for j in range(len(slots_pred[0])):
        scores = slots_pred[0][j][0]
        position = slots_pred[0][j][1]
        p0_x = width * position[0] - 0.5
        p0_y = height * position[1] - 0.5
        p1_x = width * position[2] - 0.5
        p1_y = height * position[3] - 0.5
        x1, y1, x2, y2 = p0_x, p0_y, p1_x, p1_y
        vec = np.array([p1_x - p0_x, p1_y - p0_y])
        vec = vec / np.linalg.norm(vec)
        distance =( position[0] - position[2] )**2 + ( position[1] - position[3] )**2 
        
        if VSLOT_MIN_DIST <= distance <= VSLOT_MAX_DIST:
            separating_length = LONG_SEPARATOR_LENGTH
        else:
            separating_length = SHORT_SEPARATOR_LENGTH
        
        p2_x = p0_x + height * separating_length * vec[1]
        p2_y = p0_y - width * separating_length * vec[0]
        p3_x = p1_x + height * separating_length * vec[1]
        p3_y = p1_y - width * separating_length * vec[0]

        angle1 = get_angle(p0_x, p0_y, p2_x, p2_y)
        angle2 = get_angle(p1_x, p1_y, p3_x, p3_y)
        slot_dict = {
            "points": [[x1, y1], [x2, y2]],
            "angle1": angle1,
            "angle2": angle2,
            "scores": scores.item(),
        }
        slots_list.append(slot_dict)

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
    
    slots_dict = {}
    slots_dict["slot"] = slots_list
    if len(slots_list) == 0:
        print(f"{img_name} is not be detected")
        none_dect.append(img_name)
    
    json_name = img_name.replace(".jpg", ".json")
    json_path = os.path.join(json_dir, json_name)
    with open(json_path, 'w') as f:
        json.dump(slots_dict, f)

    if display:
        for junction in junctions:
            cv2.circle(image, junction, 3, (0, 0, 255), 4)
        cv2.imshow('demo', image)
        cv2.waitKey(500)
        save_dir = Path(output_dir) / 'predictions'
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / ('%s.jpg' % img_name)
        cv2.imwrite(str(save_path), image)
    
    
if __name__ == '__main__':
    main()
