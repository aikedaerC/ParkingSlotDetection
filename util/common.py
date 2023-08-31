import numpy as np
import json
import math
import os
import scipy.io as sio
from tqdm import tqdm
# import ffmpeg # for sublinux
import subprocess


################################# for draw ################################# start 

def check_file_exists(file_path):
    if not os.path.exists(file_path):
        print(f"file '{file_path}' not found.")
        return False
    else:
        return True

def getP34(p1,p2,type_num, alpha):
    if type_num == 1:
        d = 200
    elif type_num == 2:
        d = 200
    elif type_num == 3:
        d = 200
    def rotate_clockwise(x_, y_, alpha, anticlockwise):
        # if alpha <90:
        #     alpha = 180 - alpha
        alpha = math.radians(alpha)
        if not anticlockwise:
            alpha = -alpha
        x = x_ * math.cos(alpha) - y_ * math.sin(alpha)
        y = x_ * math.sin(alpha) + y_ * math.cos(alpha)
        return np.array([x,y])

    def get_unit_vector(x,y):
        length = math.sqrt(x**2 + y**2)
        if length == 0:
            raise ValueError("Cannot compute unit vector for zero vector")
        u_x = x / length
        u_y = y / length
        return u_x, u_y

    x, y = p1[0] - p2[0], p1[1] - p2[1]
    u_x, u_y = get_unit_vector(x,y)
    rotated_vector = rotate_clockwise(u_x, u_y, alpha, True) if type_num==1 else rotate_clockwise(u_x, -u_y, alpha, False)
    p3 = rotated_vector * d + np.array(p2)
    p4 = rotated_vector * d + np.array(p1)
    # get p2p3 angle
    # 默认图像的top-left为(0,0),但比赛数据中的分离线角度的坐标原点位于bottom-right,所以这里(x,y)取(-x,-y)
    angle_radians = np.arctan2(-rotated_vector[1], -rotated_vector[0])
    # angle_degrees = np.degrees(angle_radians)
    return p3,p4,angle_radians


def img2video(img_dir,video_path_with_name=None):
    # print("s")
    video_path_with_name= img_dir if (video_path_with_name is None) else video_path_with_name
    assert not video_path_with_name.endswith("/"), "img_dir and video_path_with_name should not endwith /"
    width, height = 600, 600
    img_list = []

    for file in os.listdir(img_dir):
        if file.endswith('.jpg'):
            img_list.append(os.path.join(img_dir, file))
            
    img_list.sort()

    # Create a text file listing the input images
    with open('image_list.txt', 'w') as f:
        for img_file in img_list:
            f.write(f"file '{img_file}'\n")

    output_args = [
        '-f', 'concat',
        '-safe', '0',
        '-i', 'image_list.txt',
        '-vf', f'fps=5,scale={width}:{height}',
        f'{video_path_with_name}.mp4'
    ]

    subprocess.run(['ffmpeg'] + output_args)

    # Remove the temporary image list file
    os.remove('image_list.txt')

def getJson(root_path, json_root_path):
    file_list = os.listdir(root_path)
    for file in tqdm(file_list, desc="Processing files"):
        if file.endswith(".mat"):
        # if file == '20160725-3-647.mat':
            file_path = os.path.join(root_path, file)
        else:
            continue

        mat_data = sio.loadmat(file_path)
        marks = mat_data['marks']
        slots = mat_data['slots']
        slot_json = {"slot":[]}
        for slot in slots:
            p1_idx, p2_idx, type_num, degree = slot[0]-1, slot[1]-1, slot[2], slot[3]
            p1, p2 = marks[p1_idx], marks[p2_idx]
            p3, p4, angle_absolute = getP34(p1,p2,type_num,degree)
            p1_list, p2_list = p1.astype(float).tolist(), p2.astype(float).tolist()
            slot_dict = {
                "category": int(type_num),
                "points": [p1_list, p2_list],
                "angle1": angle_absolute,
                "angle2": angle_absolute,
                "ignore": 0.0,
                "vacant": 0.0
            }

            slot_json["slot"].append(slot_dict)
        # Write the dictionary to the JSON file

        json_file_path = os.path.join(json_root_path,file.replace(".mat",".json"))
        with open(json_file_path, 'w') as json_file:
            json.dump(slot_json, json_file, indent=4)  # indent for pretty formatting

def JsonAddAbsoluteAngle(jsonpath, new_jsonpath):
    os.makedirs(new_jsonpath, exist_ok=True)
    json_list = os.listdir(jsonpath)
    for jsonfile in tqdm(json_list, desc="JsonAddAbsoluteAngle"):
        if jsonfile.endswith(".json"):
            jsonfile_path = os.path.join(jsonpath, jsonfile)
        else:
            continue
        with open(jsonfile_path, 'r') as json_file:
            json_data = json.load(json_file)
        marks = json_data['marks']
        slots = json_data['slots']
        # print(f"{slots}   ----")
        slots = np.array(slots)
        if len(slots.shape)<=1:
            slots = np.expand_dims(slots,axis=0)
        for slot in slots:
            if len(slot) == 0:
                print(f"slot is empty:{jsonfile}")
                break
            p1_idx, p2_idx, type_num, degree = slot[0]-1, slot[1]-1, slot[2], slot[3]
            p1, p2 = marks[p1_idx][0:2], marks[p2_idx][0:2]
            p3, p4, angle_absolute = getP34(p1,p2,type_num,degree)
            if len(json_data['marks'][p1_idx])==5:
                json_data['marks'][p1_idx].append(angle_absolute)
            if len(json_data['marks'][p2_idx])==5:
                json_data['marks'][p2_idx].append(angle_absolute)
        flag = True
        for mk in json_data['marks']:
            if len(mk)<=5:
                mk.append(3.14) # for the convient of dim aligenment
            if len(mk)!=6:
                flag = False
        assert flag, f"marks has not aligened dim in {jsonfile}"

        json_file_path = os.path.join(new_jsonpath,jsonfile)
        with open(json_file_path, 'w') as json_file:
            json.dump(json_data, json_file, indent=4)  # indent for pretty formatting

def mat2Json4PointsWithAbsoluteAngle(matpath, jsonpath):
    mat_list = os.listdir(matpath)
    for mat in tqdm(mat_list, desc="mat2Json4PointsWithAbsoluteAngle"):
        if mat.endswith(".mat"):
            matfile = os.path.join(matpath, mat)
        else:
            continue
        if not check_file_exists(matfile): exit()
        mat_data = sio.loadmat(matfile)
        marks = mat_data['marks']
        slots = mat_data['slots']
        slot_json = {}
        slot_json["marks"] = marks.tolist()
        slot_json["slots"] = slots.tolist()

        for idx, slot in enumerate(slots):
            p1_idx, p2_idx, type_num, degree = slot[0]-1, slot[1]-1, slot[2], slot[3]
            p1, p2 = marks[p1_idx], marks[p2_idx]
            p3, p4, angle_absolute = getP34(p1,p2,type_num,degree)
            # add absolute to the slot list
            slot_json["slots"][idx].append(angle_absolute)
            # add another point of the separating line, if the point is not in slot, it will not get the other point, but for the convinent of indexing, just replate the same dim
            if len(slot_json["marks"][p1_idx]) == 2: # in case of sharing point
                slot_json["marks"][p1_idx].extend(p4) # 14
            if len(slot_json["marks"][p2_idx]) == 2: # in case of sharing point
                slot_json["marks"][p2_idx].extend(p3) # 23
        for idx in range(len(slot_json["marks"])):
            if len(slot_json["marks"][idx]) == 2:
                temp = slot_json["marks"][idx]*2
                slot_json["marks"][idx] = temp
                
        json_file_path = os.path.join(jsonpath, mat.replace(".mat",".json"))
        with open(json_file_path, 'w') as json_file:
            json.dump(slot_json, json_file, indent=4)  # indent for pretty formatting

def check_json_files(directory_path, oudir):
    flag = 0
    alist = []
    for filename in os.listdir(directory_path):
        if filename.endswith('.json'):
            file_path = os.path.join(directory_path, filename)
            with open(file_path, 'r') as json_file:
                data = json.load(json_file)
                
                if 'slots' in data and len(data['slots']) == 0:
                    print(f"File: {'train/imgs/'+os.path.splitext(filename)[0]+'.jpg'} - Empty 'slots' field")
                    flag+=1
                    alist.append(file_path)
                    # shutil.move('train/imgs/'+os.path.splitext(filename)[0]+'.jpg', oudir+'/imgs')
                if 'marks' in data and len(data['marks']) == 0:
                    print(f"File: {filename} - Empty 'marks' field")
    print(f"total {flag} slots is empty!")

    # for f in alist:
    #     os.remove(f)

################################# for draw   ################################# end 

if __name__ == "__main__":
    # img_dir = '/workspace/ParkingSlotDetection/data_comp/withlabelimg/predictions'
    # img2video(img_dir)

    # mat2Json4PointsWithAbsoluteAngle('/workspace/ParkingSlotDetection/data_extrap/all/mat', '/workspace/ParkingSlotDetection/data_extrap/all/json')
    
    JsonAddAbsoluteAngle('/workspace/ParkingSlotDetection/data_extra/train/json','/workspace/ParkingSlotDetection/data_extra/train/jsonWithAngle')