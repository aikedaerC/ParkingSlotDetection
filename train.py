import pprint
import torch
import argparse
from pathlib import Path
from permissive_dict import PermissiveDict as Dict
from rovver.utils import dist
from rovver.datasets.builder import build_dataloader
from rovver.models.builder import build_model
from rovver.optimization import build_optimizer, build_scheduler
from rovver.utils.config import cfg_from_file, merge_new_config, get_logger, set_random_seed
from rovver.utils.train_utils import train_model, model_fn_decorator
from tensorboardX import SummaryWriter

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


def main():
    cfg = get_config() 
    logger = get_logger(cfg.log_dir, cfg.tag)
    logger.info(pprint.pformat(cfg))

    if cfg.launcher == 'none':
        dist_train = False
    else:
        cfg.batch_size, cfg.local_rank = dist.init_dist_pytorch(
            cfg.batch_size, cfg.local_rank, backend='nccl'
        )
        cfg.data.train.batch_size = cfg.batch_size
        dist_train = True

    set_random_seed(cfg['random_seed'])

    
    if dist_train:
        total_gpus = dist.get_world_size()
        logger.info('total_batch_size: %d' % (total_gpus * cfg.batch_size))

    tb_log = SummaryWriter(log_dir=str(cfg.log_dir)) if cfg.local_rank == 0 else None

    train_set, train_loader, train_sampler = build_dataloader(
            cfg.data.train, dist=dist_train, training=True, logger=logger)

    model = build_model(cfg.model)
    if cfg.get('sync_bn', False):
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()
    
    optimizer = build_optimizer(model, cfg.optimization)

    start_epoch = it = 0
    last_epoch = -1
    if cfg.ckpt is not None:
        it, start_epoch = model.load_params_with_optimizer(cfg.ckpt, to_cpu=dist, optimizer=optimizer, logger=logger)
        last_epoch = start_epoch + 1

    model.train()
    if dist_train:
        model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[cfg.local_rank % torch.cuda.device_count()])
    logger.info(model)

    total_iters_each_epoch = len(train_loader) 
    lr_scheduler, lr_warmup_scheduler = build_scheduler(
        optimizer, total_iters_each_epoch=total_iters_each_epoch, total_epochs=cfg.epochs,
        last_epoch=last_epoch, optim_cfg=cfg.optimization
    )

    # -----------------------start training---------------------------
    logger.info('*'*20 + 'Start training' +'*'*20)
    
    train_model(
        model,
        optimizer,
        train_loader,
        model_func=model_fn_decorator(),
        lr_scheduler=lr_scheduler,
        optim_cfg=cfg.optimization,
        start_epoch=start_epoch,
        total_epochs=cfg.epochs,
        start_iter=it,
        rank=cfg.local_rank,
        tb_log=tb_log,
        ckpt_save_dir=cfg.model_dir,
        train_sampler=train_sampler,
        lr_warmup_scheduler=lr_warmup_scheduler,
        ckpt_save_interval=cfg.ckpt_save_interval,
        max_ckpt_save_num=cfg.max_ckpt_save_num,
        merge_all_iters_to_one_epoch=False
    )
    logger.info('*'*20 + 'End training' +'*'*20)
    



if __name__ == "__main__":
    main()