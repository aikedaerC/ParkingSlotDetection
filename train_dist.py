import math
import random
import torch
from torch.utils.data import DataLoader
import config
import data
import os
import time
import numpy as np
import wandb
from model import DirectionalPointDetector
## multi gpu
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp

torch.cuda.init()  # Initialize CUDA



def get_angle(angle):
    angle = angle / np.pi * 180
    mod = 360
    angle = (angle + mod) % (mod)
    if angle < 180:
        angle = angle
    else:
        angle = angle - 360
    return angle / 180 * np.pi



def generate_objective(marking_points_batch, device):
    """Get regression objective and gradient for directional point detector."""
    batch_size = len(marking_points_batch)
    objective = torch.zeros(batch_size, config.NUM_FEATURE_MAP_CHANNEL,
                            config.FEATURE_MAP_SIZE, config.FEATURE_MAP_SIZE,
                            device=device)
    gradient = torch.zeros_like(objective)
    gradient[:, 0].fill_(1.)
    for batch_idx, marking_points in enumerate(marking_points_batch):
        for marking_point in marking_points:
            col = math.floor(marking_point.x * config.FEATURE_MAP_SIZE)
            row = math.floor(marking_point.y * config.FEATURE_MAP_SIZE)
            # Confidence Regression
            objective[batch_idx, 0, row, col] = marking_point.shape
            # Makring Point Shape Regression
            objective[batch_idx, 1, row, col] = 1.0
            # Offset Regression
            objective[batch_idx, 2, row, col] = marking_point.x*16 - col
            objective[batch_idx, 3, row, col] = marking_point.y*16 - row
            # Direction Regression
            direction = marking_point.direction
            objective[batch_idx, 4, row, col] = math.cos(direction)
            objective[batch_idx, 5, row, col] = math.sin(direction)
            # Third Point Offset Regression
            # p0_x = 512 * marking_point.x - 0.5
            # p0_y = 512 * marking_point.y - 0.5
            # p2_x = p0_x - 100 * math.cos(marking_point.angle)
            # p2_y = p0_y - 100 * math.sin(marking_point.angle)
            # p2_x = (p2_x + 0.5) / 512
            # p2_y = (p2_y + 0.5) / 512
            # the above is from draw.py, I'am sure they are right to draw label
            p2_x = marking_point.x - 100 / 512 * math.cos(marking_point.angle)
            p2_y = marking_point.y - 100 / 512 * math.sin(marking_point.angle)
            # print(f"px:{marking_point.x}==>{p2_x}, py:{marking_point.y}==>{p2_y}")
            county = 10
            while p2_y >= 1:
                p2_y = marking_point.y - ((100 - county) / 512) * math.sin(marking_point.angle)
                county += 10
            
            countx = 10
            while p2_x >= 1:
                p2_x = marking_point.x - ((100 - countx) / 512) * math.sin(marking_point.angle)
                countx += 10
                
            col_third = math.floor(p2_x * config.FEATURE_MAP_SIZE)
            row_third = math.floor(p2_y * config.FEATURE_MAP_SIZE)
            objective[batch_idx, 6, row_third, col_third] = p2_x * 16 - col_third
            objective[batch_idx, 7, row_third, col_third] = p2_y * 16 - row_third
            # third point confidence
            objective[batch_idx, 8, row_third, col_third] = 1.

            # Assign Gradient
            gradient[batch_idx, 1:6, row, col].fill_(1.)
            gradient[batch_idx, 6:9, row_third, col_third].fill_(1.)
    return objective, gradient


# Define the collate function at the module level
def collate_fn(batch):
    return list(zip(*batch))

def train_detector(rank, args):
        # 初始化WandB
    if rank == 0:  # 仅在一个进程中初始化
        wandb.login(key="f18e0763419914f083eddcc562c3ce225c28b59f")
        wandb.init(project="DMPR_ANGLE", name="Final")
        os.makedirs("hweights", exist_ok=True)


    args.cuda = not args.disable_cuda and torch.cuda.is_available()
    device = torch.device('cuda:' + str(rank) if args.cuda else 'cpu')
    torch.set_grad_enabled(True)

    dp_detector = DirectionalPointDetector(3, args.depth_factor, config.NUM_FEATURE_MAP_CHANNEL)
    
    if args.detector_weights:
        print(f"rank: {rank} Loading weights: {args.detector_weights}")
        dp_detector.load_state_dict(torch.load(args.detector_weights, map_location=device))
    dp_detector = nn.SyncBatchNorm.convert_sync_batchnorm(dp_detector)  # Sync BatchNorm

    dp_detector.to(device)
    dp_detector.train()
    dp_detector = nn.parallel.DistributedDataParallel(
        dp_detector, device_ids=[rank], output_device=rank, find_unused_parameters=False
    )

    optimizer = torch.optim.Adam(dp_detector.parameters(), lr=args.lr)
    if args.optimizer_weights:
        print("Loading weights: %s" % args.optimizer_weights)
        optimizer.load_state_dict(torch.load(args.optimizer_weights, map_location=device))

    # Use DistributedSampler
    distributed_sampler = torch.utils.data.distributed.DistributedSampler(
        data.ParkingSlotDataset(args.dataset_directory),  # Pass the dataset to the sampler
        num_replicas=len(args.num_gpus),
        rank=rank
    )
    data_loader = DataLoader(
        data.ParkingSlotDataset(args.dataset_directory),
        batch_size=args.batch_size,
        shuffle=False,  # Don't shuffle with DistributedSampler
        num_workers=args.data_loading_workers,
        collate_fn=collate_fn,  # Use the collate function defined above
        sampler=distributed_sampler,  # Use the correct sampler
    )

    for epoch_idx in range(args.num_epochs):
        for iter_idx, (images, marking_points) in enumerate(data_loader):
            st = time.time()
            num_batch = len(data_loader)
            images = torch.stack(images).to(device)

            optimizer.zero_grad()
            prediction = dp_detector(images)
            objective, gradient = generate_objective(marking_points, device)
            loss_shape = (prediction[:, 0:1] - objective[:, 0:1])** 2
            loss_point = (prediction[:, 1:4] - objective[:, 1:4])** 2
            loss_direction = (prediction[:, 4:6] - objective[:, 4:6])** 2 #1- torch.cosine_similarity(prediction[:,4:5], objective[:,4:5]).unsqueeze(1)
            loss_third_point = (prediction[:, 6:9] - objective[:, 6:9])** 2
            
            loss = torch.cat((loss_shape, loss_point, loss_direction, loss_third_point), dim=1)
            loss.backward(gradient)
            optimizer.step()
            # for viual
            loss_shape = torch.sum(loss_shape * gradient[:,0:1]) / loss_shape.size(0)
            loss_point = torch.sum(loss_point * gradient[:,1:4]) / loss_point.size(0)
            loss_direction = torch.sum(loss_direction * gradient[:,4:6]) / loss_direction.size(0)
            loss_third_point = torch.sum(loss_third_point * gradient[:,6:9]) / loss_third_point.size(0)
            train_loss = loss_shape + loss_point + loss_direction + loss_third_point
            
            
            if rank == 0:
                if iter_idx % 50 == 0:
                    wandb.log({"loss_shape": loss_shape.item()})
                    wandb.log({"loss_direction": loss_direction.item()})
                    wandb.log({"loss_point": loss_point.item()})
                    wandb.log({"loss_third_point": loss_third_point.item()})
                    wandb.log({"total_loss": train_loss.item()})
                    torch.save(dp_detector.module.state_dict(), 'hweights/dp_detector.pth')
                    
                if iter_idx % 10==0:
                    current_time = time.strftime("%Y-%m-%d %H:%M:%S")
                    ep_time = time.time() - st
                    print(f"time::{current_time} |||||| epoch::{epoch_idx}/{args.num_epochs} |||||| iter::{iter_idx}/{num_batch} |||||| loss::{train_loss.item()} |||||| elapsed::{ep_time}")
                    
        if rank == 0:
            torch.save(dp_detector.module.state_dict(), 'hweights/dp_detector_%d.pth' % epoch_idx)
            torch.save(optimizer.state_dict(), 'hweights/optimizer.pth')

def init_process(rank, size, fn, args):
    """Initialize the distributed training environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend='nccl', rank=rank, world_size=size)
    fn(rank, args)

if __name__ == '__main__':
    args = config.get_parser_for_training().parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = '2,3,4,5,6'
    world_size = len(args.num_gpus)
    mp.spawn(init_process, args=(world_size, train_detector, args),
                nprocs=world_size, join=True)
