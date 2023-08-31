
import sys
import torch
import logging
import datetime
import random as pyrandom
import numpy as np
import json
import yaml
from pathlib import Path
from permissive_dict import PermissiveDict as Dict
from rovver.datasets.ps_dataset import ParkingSlotDataset
from rovver.models.detector_base import PointDetectorBase

def get_config_from_json(json_file):
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)
    # convert the dictionary to a namespace using bunch lib
    config = Dict(config_dict)
    return config, config_dict

def get_config_from_yaml(yaml_file):
    with open(yaml_file) as fp:
        config_dict = yaml.load(fp, Loader=yaml.FullLoader)
    # convert the dictionary to a namespace using bunch lib
    config = Dict(config_dict)
    return config, config_dict


def merge_new_config(config, new_config):
    if '_base_' in new_config:
        with open(new_config['_base_'], 'r') as f:
            try:
                yaml_config = yaml.load(f, Loader=yaml.FullLoader)
            except:
                yaml_config = yaml.load(f, Loader=yaml.FullLoader)
            print(yaml_config)
        config.update(Dict(yaml_config))

    for key, val in new_config.items():
        if not isinstance(val, dict):
            config[key] = val
            continue
        if key not in config:
            config[key] = Dict()
        merge_new_config(config[key], val)

    return config

def cfg_from_file(config_file):
    if config_file.endswith('json'):
        new_config, _ = get_config_from_json(config_file)
    elif config_file.endswith('yaml'):
        new_config, _ = get_config_from_yaml(config_file)
    else:
        raise Exception("Only .json and .yaml are supported!")

    config = Dict()
    merge_new_config(config=config, new_config=new_config)
    return config

################################# for config ################################# end 



def set_random_seed(seed=3):
    pyrandom.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_logger(logdir, name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    ts = str(datetime.datetime.now()).split(".")[0].replace(" ", "_")
    ts = ts.replace(":", "_").replace("-", "_")
    file_path = Path(logdir) / "run_{}.log".format(ts)
    file_hdlr = logging.FileHandler(str(file_path))
    file_hdlr.setFormatter(formatter)

    strm_hdlr = logging.StreamHandler(sys.stdout)
    strm_hdlr.setFormatter(formatter)

    logger.addHandler(file_hdlr)
    logger.addHandler(strm_hdlr)
    return logger
