import math
import random
import torch
from torch.utils.data import DataLoader
import config
import data
import os
from tqdm import tqdm
import numpy as np
import util
import wandb
from model import DirectionalPointDetector
## multi gpu
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp

torch.cuda.init()  # Initialize CUDA


def plot_prediction(logger, image, marking_points, prediction):
    """Plot the ground truth and prediction of a random sample in a batch."""
    rand_sample = random.randint(0, image.size(0)-1)
    sampled_image = util.tensor2im(image[rand_sample])
    logger.plot_marking_points(sampled_image, marking_points[rand_sample],
                               win_name='gt_marking_points')
    sampled_image = util.tensor2im(image[rand_sample])
    pred_points = data.get_predicted_points(prediction[rand_sample], 0.01)
    if pred_points:
        logger.plot_marking_points(sampled_image,
                                   list(list(zip(*pred_points))[1]),
                                   win_name='pred_marking_points')


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
            objective[batch_idx, 0, row, col] = 1.
            # Makring Point Shape Regression
            objective[batch_idx, 1, row, col] = marking_point.shape
            # Offset Regression
            objective[batch_idx, 2, row, col] = marking_point.x*16 - col
            objective[batch_idx, 3, row, col] = marking_point.y*16 - row
            # Direction Regression
            direction = marking_point.direction
            objective[batch_idx, 4, row, col] = math.cos(direction)
            objective[batch_idx, 5, row, col] = math.sin(direction)
            # Assign Gradient
            gradient[batch_idx, 1:6, row, col].fill_(1.)
    return objective, gradient

# Define the collate function at the module level
def collate_fn(batch):
    return list(zip(*batch))

def train_detector(rank, args):
        # 初始化WandB
    if rank == 0:  # 仅在一个进程中初始化
        wandb.login(key="f18e0763419914f083eddcc562c3ce225c28b59f")
        wandb.init(project="DMPR", name="DistributedTraining")

    args.cuda = not args.disable_cuda and torch.cuda.is_available()
    device = torch.device('cuda:' + str(rank) if args.cuda else 'cpu')
    torch.set_grad_enabled(True)

    dp_detector = DirectionalPointDetector(
        3, args.depth_factor, config.NUM_FEATURE_MAP_CHANNEL)
    if args.detector_weights:
        print(f"rank: {rank} Loading weights: {args.detector_weights}")
        dp_detector.load_state_dict(torch.load(args.detector_weights, map_location=device))
    dp_detector.to(device)
    dp_detector.train()

    dp_detector = nn.SyncBatchNorm.convert_sync_batchnorm(dp_detector)  # Sync BatchNorm

    dp_detector = nn.parallel.DistributedDataParallel(
        dp_detector, device_ids=[rank], output_device=rank, find_unused_parameters=True
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
    min_loss = 0.004322
    for epoch_idx in tqdm(range(args.num_epochs)):
        for iter_idx, (images, marking_points) in enumerate(data_loader):
            images = torch.stack(images).to(device)

            optimizer.zero_grad()
            prediction = dp_detector(images)
            objective, gradient = generate_objective(marking_points, device)
            loss = (prediction - objective) ** 2
            loss.backward(gradient)
            optimizer.step()

            train_loss = torch.sum(loss * gradient).item() / loss.size(0)

            if rank == 0:
                wandb.log({"train_iter_loss": train_loss}, step=iter_idx)
                if train_loss < min_loss:
                    min_loss = train_loss
                    torch.save(dp_detector.module.state_dict(),
                            f"weights/iter{iter_idx}_loss{min_loss}.pth")

        if rank == 0:
            wandb.log({"train_epoch_loss": train_loss}, step=epoch_idx)


        if rank == 0:
            torch.save(dp_detector.module.state_dict(),
                       'weights/dp_detector_%d.pth' % epoch_idx)
            torch.save(optimizer.state_dict(), 'weights/optimizer.pth')

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

# python -u train_dist.py --detector_weights /home/tlgr/ZTY/bk/DMPR-PS-master/weights/model_123none.pth