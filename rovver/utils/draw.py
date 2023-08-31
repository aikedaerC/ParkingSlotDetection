import cv2
import json
import math 
import argparse
import os
import scipy.io as sio
import sys
sys.path.append(r"F:\ubunut_desktop_back\ParkingSlotDetection\rovver")
from tqdm import tqdm
from rovver.utils.common import check_file_exists, getP34

# in powershell 
# $env:PYTHONPATH="C:\\Users\\aikedaer\\Desktop\\parkingslotcompetition\\v0" 


parser = argparse.ArgumentParser(description='Process images and marks')
parser.add_argument('--mode',              default='aug',             help='single or many or pp, pp only for data_extrap, aug')
parser.add_argument('--data_type',         default='data_extra',        help='data_comp or data_extrap')
parser.add_argument('--save',              default='yes',               help='yes or no, care for the ouput dir')
parser.add_argument('--interactivity',     default='yes',              help='yes or no, press esc for next img')
args = parser.parse_args()
# 
root_path = r'F:\ubunut_desktop_back\ParkingSlotDetection\data_extra\train\aug\train'


if args.mode == 'single':
    if args.data_type == 'data_extra':
        img_path = os.path.join(root_path,'v0/data_extra/train/imgs/20160725-3-1.jpg') #20160725-3-1.json
        json_path = os.path.join(root_path,'v0/data_extra/train/json/20160725-3-1.json')
    elif args.data_type == 'data_comp':
        img_path = os.path.join(root_path,'v0/data_comp/train/imgs/20160725-3-14.jpg')
        json_path = os.path.join(root_path, 'v0/data_comp/train/json/20160725-3-14.json')

    # Load the fisheye image
    fisheye_image = cv2.imread(img_path)

    # JSON data
    json_file_path = json_path
    with open(json_file_path, 'r') as json_file:
        json_data = json.load(json_file)
    # print(json_data)
    # data_extra
    if args.data_type == 'data_extra':
        for mark in tqdm(json_data["marks"], desc="data_extra"):
            x1, y1, x2, y2, _ = mark
            direction = math.atan2((y1 - y2),(x1 - x2))
            # print(direction)
            cv2.line(fisheye_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.circle(fisheye_image, (int(x1), int(y1)), 5, (255, 0, 0), -1)  # Draw a blue circle at mark1
            cv2.circle(fisheye_image, (int(x2), int(y2)), 5, (0, 0, 255), -1)  # Draw a red circle at mark2
    elif args.data_type == 'data_comp':
        for slot in tqdm(json_data["slot"], desc="data_comp"):
            points = slot["points"]
            angle = slot["angle1"] 
            
            x1, y1 = points[0]
            x2, y2 = points[1]
            
            length = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            dx = length * math.cos(angle)
            dy = length * math.sin(angle)
            
            x3 = x2 - dx
            y3 = y2 - dy
            
            x4 = x1 - dx
            y4 = y1 - dy
            
            cv2.line(fisheye_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.line(fisheye_image, (int(x2), int(y2)), (int(x3), int(y3)), (0, 255, 0), 2)
            cv2.line(fisheye_image, (int(x3), int(y3)), (int(x4), int(y4)), (0, 255, 0), 2)
            cv2.line(fisheye_image, (int(x4), int(y4)), (int(x1), int(y1)), (0, 255, 0), 2)

    # Initialize coordinate display variables
    display_tooltip = False
    tooltip_text = ""
    # Mouse callback function
    def mouse_callback(event, x, y, flags, param):
        global display_tooltip, tooltip_text
        if event == cv2.EVENT_MOUSEMOVE:
            tooltip_text = f"Pixel: ({x}, {y})"
            display_tooltip = True
        else:
            display_tooltip = False
    # Create a window and set mouse callback function
    cv2.namedWindow("Fisheye Image with Lines and Marks")
    cv2.setMouseCallback("Fisheye Image with Lines and Marks", mouse_callback)
    # Display the modified image with args.interactivity
    while True:
        image_to_display = fisheye_image.copy()

        if display_tooltip:
            cv2.putText(image_to_display, tooltip_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Fisheye Image with Lines and Marks", image_to_display)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # Press 'Esc' to exit
            break
    # cv2.destroyAllWindows()

elif args.mode == 'many':
    if args.data_type == 'data_extrap':
        img_dir = os.path.join(root_path,'/workspace/ParkingSlotDetection/data_extrap/all/imgs') #20160725-3-1.json
        json_dir = os.path.join(root_path,'/workspace/ParkingSlotDetection/data_extrap/all/json')
    elif args.data_type == 'data_comp':
        img_dir = os.path.join(root_path,'imgs')
        json_dir = os.path.join(root_path, 'json')
    withlabel_path = img_dir.replace("imgs", "withlabel")
    os.makedirs(withlabel_path, exist_ok=True)

    img_list = os.listdir(img_dir)
    for img in tqdm(img_list, desc="Img list"):
        if img.endswith(".jpg"):
            img_path = os.path.join(img_dir, img)
        else:
            continue
        fisheye_image = cv2.imread(img_path)
        json_path = img_path.replace("imgs", "json").replace('.jpg', '.json')
        if not check_file_exists(json_path): exit()
        with open(json_path, 'r') as json_file:
            json_data = json.load(json_file)
        # print(json_data)
        if args.data_type == 'data_extrap':
            for slot in json_data["slots"]:
                try:
                    point0 = json_data["marks"][slot[0]-1]
                    point1 = json_data["marks"][slot[1]-1]
                    cv2.line(fisheye_image, (int(point0[0]), int(point0[1])), (int(point1[0]), int(point1[1])), (0, 0, 255), 2)
                except:
                    print(slot)
            for mark in json_data["marks"]:
                if len(mark) != 4:
                    continue
                x1, y1, x2, y2 = mark
                # direction = math.atan2((y1 - y2),(x1 - x2))
                # print(direction)
                cv2.line(fisheye_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.circle(fisheye_image, (int(x1), int(y1)), 5, (255, 0, 0), -1)  # Draw a blue circle at mark1
                cv2.circle(fisheye_image, (int(x2), int(y2)), 5, (0, 0, 255), -1)  # Draw a red circle at mark2
        elif args.data_type == 'data_comp':
            for idx, slot in enumerate(json_data["slot"]):
                points = slot["points"]
                x1, y1 = points[0]
                x2, y2 = points[1]
                
                # Convert angles to degrees for drawing
                angle1_deg = slot["angle1"] # * 180 / 3.141592653589793
                angle2_deg = slot["angle2"] # * 180 / 3.141592653589793
                
                cv2.line(fisheye_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.circle(fisheye_image, (int(x1), int(y1)), 5, (255, 0, 0), -1)  # Draw a blue circle at point1
                cv2.circle(fisheye_image, (int(x2), int(y2)), 5, (0, 0, 255), -1)  # Draw a red circle at point2
                x3 = x1 - 100 * math.cos(angle1_deg)
                y3 = y1 - 100 * math.sin(angle1_deg)
                x4 = x2 - 100 * math.cos(angle2_deg)
                y4 = y2 - 100 * math.sin(angle2_deg)
                cv2.line(fisheye_image, (int(x1), int(y1)), (int(x3), int(y3)), (0, 255, 0), 2)
                cv2.line(fisheye_image, (int(x4), int(y4)), (int(x2), int(y2)), (0, 255, 0), 2)

                # Display angle information
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(fisheye_image, f"Angle {idx}-1: {angle1_deg:.2f} degrees", (int(x1) + 10*idx, int(y1) - 10*idx), font, 0.5, (0, 255, 0), 2)
                cv2.putText(fisheye_image, f"Angle {idx}-2: {angle2_deg:.2f} degrees", (int(x2) + 10*idx, int(y2) - 10*idx), font, 0.5, (0, 255, 0), 2)


        if args.interactivity == 'yes':
            display_tooltip = False
            tooltip_text = ""
            # Mouse callback function
            def mouse_callback(event, x, y, flags, param):
                global display_tooltip, tooltip_text

                if event == cv2.EVENT_MOUSEMOVE:
                    tooltip_text = f"Pixel: ({x}, {y})"
                    display_tooltip = True
                else:
                    display_tooltip = False
            # Display the modified image with args.interactivity
            cv2.namedWindow("Fisheye Image with Lines and Marks")
            cv2.setMouseCallback("Fisheye Image with Lines and Marks", mouse_callback)

            while True:
                image_to_display = fisheye_image.copy()

                if display_tooltip:
                    cv2.putText(image_to_display, tooltip_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                cv2.imshow("Fisheye Image with Lines and Marks", image_to_display)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # Press 'Esc' to exit
                    break
            # cv2.destroyAllWindows()
        else:
            if args.save == "yes":
                withlabel_file = os.path.join(withlabel_path, img)
                cv2.imwrite(withlabel_file, fisheye_image)
            else:
                cv2.imshow("Fisheye Image with Lines and Marks", fisheye_image)
                cv2.waitKey(200)  # 图像显示时间，以毫秒为单位

elif args.mode == 'pp':
    img_dir = os.path.join(root_path,'v0/data_extrap/train')
    assert not img_dir.endswith("/"), "img_dir should not endwith /"
    withlabel_path = img_dir.replace("train","withlabel")
    os.makedirs(withlabel_path, exist_ok=True) # 这将在指定的路径上创建目录，如果目录已经存在，则不会引发错误
    # Create a window and set mouse callback function
    # rui = ['p2_img54_0000.jpg', 'p2_img54_0018.jpg', 'p2_img54_0054.jpg', 'p2_img54_0060.jpg', 'p2_img54_0066.jpg', 'p2_img54_0078.jpg', 'p2_img54_0126.jpg', 'p2_img54_0138.jpg', 'p2_img54_0150.jpg', 'p2_img54_0162.jpg', 'p2_img54_0174.jpg', 'p2_img54_0192.jpg', 'p2_img54_0198.jpg', 'p2_img54_0204.jpg', 'p2_img54_0228.jpg', 'p2_img54_0240.jpg', 'p2_img54_0252.jpg', 'p2_img54_0312.jpg', 'p2_img54_0318.jpg', 'p2_img54_0324.jpg', 'p2_img54_0336.jpg', 'p2_img54_0342.jpg', 'p2_img54_0348.jpg', 'p2_img54_0360.jpg', 'p2_img54_0366.jpg', 'p2_img54_0372.jpg', 'p2_img54_0384.jpg', 'p2_img54_0390.jpg', 'p2_img54_0396.jpg', 'p2_img54_0408.jpg', 'p2_img54_0414.jpg', 'p2_img54_0420.jpg', 'p2_img54_0432.jpg', 'p2_img54_0438.jpg', 'p2_img54_0444.jpg', 'p2_img54_0456.jpg', 'p2_img54_0462.jpg', 'p2_img54_0474.jpg', 'p2_img54_0486.jpg', 'p2_img54_0492.jpg', 'p2_img54_0498.jpg', 'p2_img54_0510.jpg', 'p2_img54_0516.jpg', 'p2_img54_0522.jpg', 'p2_img54_0534.jpg', 'p2_img54_0540.jpg', 'p2_img54_0546.jpg', 'p2_img54_0558.jpg', 'p2_img54_0618.jpg', 'p2_img54_0624.jpg', 'p2_img54_0636.jpg', 'p2_img55_0132.jpg', 'p2_img55_0156.jpg']
    # dun = ['p2_img116_0000.jpg', 'p2_img116_0060.jpg', 'p2_img116_0084.jpg', 'p2_img116_0114.jpg', 'p2_img116_0198.jpg', 'p2_img116_0216.jpg', 'p2_img116_0270.jpg', 'p2_img116_0348.jpg', 'p2_img116_0384.jpg', 'p2_img116_0426.jpg', 'p2_img117_0036.jpg', 'p2_img117_0072.jpg', 'p2_img117_0120.jpg', 'p2_img117_0240.jpg', 'p2_img117_0306.jpg', 'p2_img117_0390.jpg', 'p2_img117_0438.jpg', 'p2_img117_0486.jpg', 'p2_img117_0516.jpg', 'p2_img117_0594.jpg', 'p2_img117_0666.jpg', 'p2_img117_0708.jpg', 'p2_img118_0042.jpg', 'p2_img118_0078.jpg', 'p2_img118_0114.jpg', 'p2_img118_0258.jpg', 'p2_img118_0288.jpg', 'p2_img118_0312.jpg', 'p2_img118_0360.jpg', 'p2_img118_0378.jpg', 'p2_img118_0420.jpg', 'p2_img118_0480.jpg', 'p2_img118_0510.jpg', 'p2_img118_0534.jpg', 'p2_img118_0630.jpg', 'p2_img118_0654.jpg', 'p2_img118_0690.jpg', 'p2_img118_0780.jpg', 'p2_img118_0834.jpg', 'p2_img118_0864.jpg', 'p2_img119_0000.jpg', 'p2_img119_0024.jpg', 'p2_img119_0060.jpg', 'p2_img119_0120.jpg', 'p2_img119_0150.jpg', 'p2_img119_0180.jpg', 'p2_img119_0240.jpg', 'p2_img119_0306.jpg', 'p2_img119_0348.jpg', 'p2_img119_0408.jpg']
    # zhi = ['20160816-7-1415.jpg', '20160816-7-1446.jpg', '20160816-8-1074.jpg', '20160816-8-1090.jpg', '20160816-8-1100.jpg', '20160816-8-1544.jpg', '20161019-1-385.jpg', '20161019-1-388.jpg', '20161019-1-389.jpg', '20161019-1-393.jpg', '20161019-1-395.jpg', '20161019-1-417.jpg', '20161019-1-420.jpg', '20161019-1-426.jpg', '20161019-2-14.jpg', '20161019-2-495.jpg', '20161019-2-520.jpg', '20161019-2-522.jpg', '20161019-2-56.jpg', '20161019-2-58.jpg', '20161019-2-623.jpg', '20161019-2-626.jpg', '20161019-2-635.jpg', '20161019-2-653.jpg', '20161019-2-692.jpg', '20161019-2-698.jpg', '20161019-2-908.jpg', '20161102-146.jpg', '20161102-150.jpg', '20161102-158.jpg', '20161102-161.jpg', '20161102-163.jpg', '20161102-166.jpg', '20161102-171.jpg', '20161102-180.jpg', '20161102-200.jpg', '20161102-203.jpg', '20161102-219.jpg', '20161102-221.jpg', '20161102-224.jpg', '20161102-227.jpg', '20161102-231.jpg', '20161102-238.jpg', '20161102-245.jpg', '20161102-257.jpg', '20161102-262.jpg', '20161102-274.jpg', '20161102-278.jpg', '20161102-301.jpg', '20161102-307.jpg', '20161102-309.jpg', '20161102-312.jpg', '20161102-316.jpg', '20161102-324.jpg', '20161102-331.jpg', '20161102-357.jpg', '20161102-368.jpg', '20161102-372.jpg', '20161102-375.jpg', '20161102-385.jpg', '20161102-388.jpg', '20161102-392.jpg', '20161102-402.jpg', '20161102-411.jpg', '20161102-417.jpg', '20161102-425.jpg', '20161102-428.jpg', '20161102-432.jpg', '20161102-440.jpg', '20161102-471.jpg', '20161102-473.jpg', '20161102-475.jpg']
    img_list = os.listdir(img_dir)
    for img in tqdm(img_list, desc="Img list"):
        # if img in rui:
        if img.endswith(".jpg"):
            img_path = os.path.join(img_dir, img)
            
        else:
            continue
        # print(img_path)
        fisheye_image = cv2.imread(img_path)
        mat_path = img_path.replace('.jpg', '.mat')
        if not check_file_exists(mat_path): exit()
        mat_data = sio.loadmat(mat_path)
        marks = mat_data['marks']
        slots = mat_data['slots']

        # print(f"marks: {marks}")
        # print(f"slots: {slots}")
        for slot in slots:
            p1_idx, p2_idx, type_num, degree = slot[0]-1, slot[1]-1, slot[2], slot[3]
            p1, p2 = marks[p1_idx], marks[p2_idx]
            p3, p4, angle_absolute = getP34(p1,p2,type_num,degree)
            # print(f"angle: {angle_absolute}")
            # print()
            # plot p1p2
            cv2.line(fisheye_image, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 255, 0), 2)
            cv2.circle(fisheye_image, (int(p1[0]), int(p1[1])), 5, (255, 0, 0), -1)  # Draw a blue circle at point1
            cv2.circle(fisheye_image, (int(p2[0]), int(p2[1])), 5, (255, 0, 0), -1)  # Draw a red circle at point2 
            # plot p2p3
            cv2.line(fisheye_image, (int(p2[0]), int(p2[1])), (int(p3[0]), int(p3[1])), (0, 255, 0), 2)
            cv2.circle(fisheye_image, (int(p3[0]), int(p3[1])), 5, (0, 0, 255), -1)  # Draw a red circle at point2 
            # plot p1p4
            cv2.line(fisheye_image, (int(p1[0]), int(p1[1])), (int(p4[0]), int(p4[1])), (0, 255, 0), 2)
            cv2.circle(fisheye_image, (int(p4[0]), int(p4[1])), 5, (0, 0, 255), -1)  # Draw a red circle at point2 

        if args.interactivity == 'yes':
            display_tooltip = False
            tooltip_text = ""
            # Mouse callback function
            def mouse_callback(event, x, y, flags, param):
                global display_tooltip, tooltip_text

                if event == cv2.EVENT_MOUSEMOVE:
                    tooltip_text = f"Pixel: ({x}, {y})"
                    display_tooltip = True
                else:
                    display_tooltip = False
            # Display the modified image with args.interactivity
            cv2.namedWindow("Fisheye Image with Lines and Marks")
            cv2.setMouseCallback("Fisheye Image with Lines and Marks", mouse_callback)

            while True:
                image_to_display = fisheye_image.copy()

                if display_tooltip:
                    cv2.putText(image_to_display, tooltip_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                cv2.imshow("Fisheye Image with Lines and Marks", image_to_display)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # Press 'Esc' to exit
                    break
            # cv2.destroyAllWindows()
        else:
            if args.save == "yes":
                withlabel_file = os.path.join(withlabel_path, img)
                cv2.imwrite(withlabel_file, fisheye_image)
            else:
                cv2.imshow("Fisheye Image with Lines and Marks", fisheye_image)
                cv2.waitKey(300)  # 图像显示时间，以毫秒为单位



elif args.mode == 'aug':
    img_dir = root_path
    assert not img_dir.endswith("/"), "img_dir should not endwith /"
    withlabel_path = os.path.join(root_path,"withlabel")
    os.makedirs(withlabel_path, exist_ok=True) # 这将在指定的路径上创建目录，如果目录已经存在，则不会引发错误
    # Create a window and set mouse callback function
    # rui = ['p2_img54_0000.jpg', 'p2_img54_0018.jpg', 'p2_img54_0054.jpg', 'p2_img54_0060.jpg', 'p2_img54_0066.jpg', 'p2_img54_0078.jpg', 'p2_img54_0126.jpg', 'p2_img54_0138.jpg', 'p2_img54_0150.jpg', 'p2_img54_0162.jpg', 'p2_img54_0174.jpg', 'p2_img54_0192.jpg', 'p2_img54_0198.jpg', 'p2_img54_0204.jpg', 'p2_img54_0228.jpg', 'p2_img54_0240.jpg', 'p2_img54_0252.jpg', 'p2_img54_0312.jpg', 'p2_img54_0318.jpg', 'p2_img54_0324.jpg', 'p2_img54_0336.jpg', 'p2_img54_0342.jpg', 'p2_img54_0348.jpg', 'p2_img54_0360.jpg', 'p2_img54_0366.jpg', 'p2_img54_0372.jpg', 'p2_img54_0384.jpg', 'p2_img54_0390.jpg', 'p2_img54_0396.jpg', 'p2_img54_0408.jpg', 'p2_img54_0414.jpg', 'p2_img54_0420.jpg', 'p2_img54_0432.jpg', 'p2_img54_0438.jpg', 'p2_img54_0444.jpg', 'p2_img54_0456.jpg', 'p2_img54_0462.jpg', 'p2_img54_0474.jpg', 'p2_img54_0486.jpg', 'p2_img54_0492.jpg', 'p2_img54_0498.jpg', 'p2_img54_0510.jpg', 'p2_img54_0516.jpg', 'p2_img54_0522.jpg', 'p2_img54_0534.jpg', 'p2_img54_0540.jpg', 'p2_img54_0546.jpg', 'p2_img54_0558.jpg', 'p2_img54_0618.jpg', 'p2_img54_0624.jpg', 'p2_img54_0636.jpg', 'p2_img55_0132.jpg', 'p2_img55_0156.jpg']
    # dun = ['p2_img116_0000.jpg', 'p2_img116_0060.jpg', 'p2_img116_0084.jpg', 'p2_img116_0114.jpg', 'p2_img116_0198.jpg', 'p2_img116_0216.jpg', 'p2_img116_0270.jpg', 'p2_img116_0348.jpg', 'p2_img116_0384.jpg', 'p2_img116_0426.jpg', 'p2_img117_0036.jpg', 'p2_img117_0072.jpg', 'p2_img117_0120.jpg', 'p2_img117_0240.jpg', 'p2_img117_0306.jpg', 'p2_img117_0390.jpg', 'p2_img117_0438.jpg', 'p2_img117_0486.jpg', 'p2_img117_0516.jpg', 'p2_img117_0594.jpg', 'p2_img117_0666.jpg', 'p2_img117_0708.jpg', 'p2_img118_0042.jpg', 'p2_img118_0078.jpg', 'p2_img118_0114.jpg', 'p2_img118_0258.jpg', 'p2_img118_0288.jpg', 'p2_img118_0312.jpg', 'p2_img118_0360.jpg', 'p2_img118_0378.jpg', 'p2_img118_0420.jpg', 'p2_img118_0480.jpg', 'p2_img118_0510.jpg', 'p2_img118_0534.jpg', 'p2_img118_0630.jpg', 'p2_img118_0654.jpg', 'p2_img118_0690.jpg', 'p2_img118_0780.jpg', 'p2_img118_0834.jpg', 'p2_img118_0864.jpg', 'p2_img119_0000.jpg', 'p2_img119_0024.jpg', 'p2_img119_0060.jpg', 'p2_img119_0120.jpg', 'p2_img119_0150.jpg', 'p2_img119_0180.jpg', 'p2_img119_0240.jpg', 'p2_img119_0306.jpg', 'p2_img119_0348.jpg', 'p2_img119_0408.jpg']
    # zhi = ['20160816-7-1415.jpg', '20160816-7-1446.jpg', '20160816-8-1074.jpg', '20160816-8-1090.jpg', '20160816-8-1100.jpg', '20160816-8-1544.jpg', '20161019-1-385.jpg', '20161019-1-388.jpg', '20161019-1-389.jpg', '20161019-1-393.jpg', '20161019-1-395.jpg', '20161019-1-417.jpg', '20161019-1-420.jpg', '20161019-1-426.jpg', '20161019-2-14.jpg', '20161019-2-495.jpg', '20161019-2-520.jpg', '20161019-2-522.jpg', '20161019-2-56.jpg', '20161019-2-58.jpg', '20161019-2-623.jpg', '20161019-2-626.jpg', '20161019-2-635.jpg', '20161019-2-653.jpg', '20161019-2-692.jpg', '20161019-2-698.jpg', '20161019-2-908.jpg', '20161102-146.jpg', '20161102-150.jpg', '20161102-158.jpg', '20161102-161.jpg', '20161102-163.jpg', '20161102-166.jpg', '20161102-171.jpg', '20161102-180.jpg', '20161102-200.jpg', '20161102-203.jpg', '20161102-219.jpg', '20161102-221.jpg', '20161102-224.jpg', '20161102-227.jpg', '20161102-231.jpg', '20161102-238.jpg', '20161102-245.jpg', '20161102-257.jpg', '20161102-262.jpg', '20161102-274.jpg', '20161102-278.jpg', '20161102-301.jpg', '20161102-307.jpg', '20161102-309.jpg', '20161102-312.jpg', '20161102-316.jpg', '20161102-324.jpg', '20161102-331.jpg', '20161102-357.jpg', '20161102-368.jpg', '20161102-372.jpg', '20161102-375.jpg', '20161102-385.jpg', '20161102-388.jpg', '20161102-392.jpg', '20161102-402.jpg', '20161102-411.jpg', '20161102-417.jpg', '20161102-425.jpg', '20161102-428.jpg', '20161102-432.jpg', '20161102-440.jpg', '20161102-471.jpg', '20161102-473.jpg', '20161102-475.jpg']
    from collections import namedtuple
    import numpy as np
    MarkingPoint = namedtuple('MarkingPoint', ['x', 'y', 'direction', 'shape', 'angle'])
    sample_names = []
    for file in os.listdir(img_dir):
        if file.endswith(".json"):
            sample_names.append(os.path.splitext(file)[0])
    print(img_dir)
    for name in tqdm(sample_names):
        print(f"name: {name}")
        image = cv2.imread(os.path.join(img_dir, name + '.jpg'))
        # with open(os.path.join(jsonWithAngle_dir, name + '.json'), 'r') as json_file:
        #     json_data = json.load(json_file)
        #     for slot in json_data["marks"]:
        #         marking_point = MarkingPoint(*slot)
        #         if marking_point.angle == 3.14:
        #             continue
        #         p0_x = marking_point.x
        #         p0_y = marking_point.y
        #         p0_x_sp = marking_point.x_sp
        #         p0_y_sp = marking_point.y_sp
        #         cv2.circle(image, (int(p0_x), int(p0_y)), 5, (255, 0, 0), -1)
        #         cv2.circle(image, (int(p0_x_sp), int(p0_y_sp)), 5, (0, 255, 0), -1)

        #         # p2_x = p0_x - 100 * math.cos(marking_point.angle)
        #         # p2_y = p0_y - 100 * math.sin(marking_point.angle)
        #         # print(f"marking_point: {p0_x, p0_y, p2_x, p2_y, marking_point.direction, marking_point.shape, marking_point.angle, marking_point.angle/np.pi * 180}")

        #         cv2.circle(image, (int(p0_x_sp), int(p0_y_sp)), 5, (0, 0, 255), -1)

        #         cv2.line(image, (int(p0_x), int(p0_y)), (int(p0_x_sp), int(p0_y_sp)), (0, 255, 0), 2)

        with open(os.path.join(img_dir, name + '.json'), 'r') as file:
            for label in json.load(file):
                marking_point = MarkingPoint(*label)
                if marking_point.angle == 3.14:
                    continue
                p0_x = 512 * marking_point.x - 0.5
                p0_y = 512 * marking_point.y - 0.5

                cv2.circle(image, (int(p0_x), int(p0_y)), 5, (255, 0, 0), -1)
                # cv2.circle(image, (int(p0_x_sp), int(p0_y_sp)), 5, (0, 255, 0), -1)

                p2_x = p0_x - 100 * math.cos(marking_point.angle)
                p2_y = p0_y - 100 * math.sin(marking_point.angle)
                p2_x = (p2_x + 0.5) / 512
                p2_y = (p2_y + 0.5) / 512
                p2_x = 512 * p2_x - 0.5
                p2_y = 512 * p2_y - 0.5
                print(f"marking_point: {p0_x, p0_y, p2_x, p2_y, marking_point.direction, marking_point.shape, marking_point.angle, marking_point.angle/np.pi * 180}")

                cv2.circle(image, (int(p2_x), int(p2_y)), 5, (0, 0, 255), -1)

                cv2.line(image, (int(p0_x), int(p0_y)), (int(p2_x), int(p2_y)), (0, 255, 0), 2)
            
            
        display_tooltip = False
        tooltip_text = ""
        # Mouse callback function
        def mouse_callback(event, x, y, flags, param):
            global display_tooltip, tooltip_text

            if event == cv2.EVENT_MOUSEMOVE:
                tooltip_text = f"Pixel: ({x}, {y})"
                display_tooltip = True
            else:
                display_tooltip = False
        # Display the modified image with args.interactivity
        cv2.namedWindow("Fisheye Image with Lines and Marks")
        cv2.setMouseCallback("Fisheye Image with Lines and Marks", mouse_callback)

        while True:
            image_to_display = image.copy()

            if display_tooltip:
                cv2.putText(image_to_display, tooltip_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow("Fisheye Image with Lines and Marks", image_to_display)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # Press 'Esc' to exit
                break