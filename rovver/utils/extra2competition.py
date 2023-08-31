import os
from rovver.utils.common import getJson, img2video
###################################################################### 
root_path = '/mnt/c/users/aikedaer/Desktop/parkingslotcompetition/v0/data_extrap/train/'
json_root_path = root_path.replace("train/", "train_json")
os.makedirs(json_root_path, exist_ok=True)


if __name__ == "__main__":
    # using sublinux
    getJson()
    # using sublinux
    # img2video('/mnt/c/users/aikedaer/Desktop/parkingslotcompetition/v0/data_extrap/withlabel')