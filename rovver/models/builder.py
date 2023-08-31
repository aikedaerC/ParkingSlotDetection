from rovver.utils.registry import build_from_cfg, Registry
from torch import nn

# point detector
POINT_DETECTOR = Registry('point_detector')


def build(cfg, registry, default_args=None):
    if isinstance(cfg, list):
        modules = [build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg]
        # print(f"modules: {modules}")
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)


def build_model(cfg, train_cfg=None, test_cfg=None):
    return build(cfg, POINT_DETECTOR)

if __name__ == "__main__":
    from rovver.utils.config import cfg_from_file, merge_new_config
    import argparse
    from pathlib import Path
    from permissive_dict import PermissiveDict as Dict

    def get_args():
        argparser = argparse.ArgumentParser(description=__doc__)
        argparser.add_argument( '-c', '--cfg',      metavar='C',           default="yamls/ps_gat.yaml",                                help='The Configuration file')
        argparser.add_argument( '-s', '--seed',     default=100,           type=int,                                                    help='The random seed')
        argparser.add_argument( '-m', '--ckpt',     type=str,              default="cache/check_pretrained/checkpoint_epoch_200.pth",   help='The model path')
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
        config.tag = Path(config_file).stem # 文件名不包含路径和后缀
        config.cache_dir = Path('cache') / config.tag / str(config.random_seed)
        config.model_dir = config.cache_dir / 'models'
        config.log_dir = config.cache_dir / 'logs'
        config.output_dir = config.cache_dir / 'output'
        
        # create the experiments dirs
        config.cache_dir.resolve().mkdir(parents=True, exist_ok=True) 
        config.model_dir.resolve().mkdir(exist_ok=True)
        config.log_dir.resolve().mkdir(exist_ok=True)
        config.output_dir.resolve().mkdir(exist_ok=True)
        
        cfg = Dict()
        merge_new_config(config=cfg, new_config=config)
        return cfg

    cfg = get_config()

    build_model(cfg.model)