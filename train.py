"""Train directional marking point detector."""
import math
import random
import torch
from torch.utils.data import DataLoader
import config
import data
import util
import os
import numpy as np
from model import DirectionalPointDetector


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
            objective[batch_idx, 1, row, col] = 1.
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


def train_detector(args):
    """Train directional point detector."""
    args.cuda = not args.disable_cuda and torch.cuda.is_available()
    device = torch.device('cuda:' + str(args.gpu_id) if args.cuda else 'cpu')
    torch.set_grad_enabled(True)

    dp_detector = DirectionalPointDetector(
        3, args.depth_factor, config.NUM_FEATURE_MAP_CHANNEL).to(device)
    if args.detector_weights:
        print("Loading weights: %s" % args.detector_weights)
        dp_detector.load_state_dict(torch.load(args.detector_weights))
    dp_detector.train()

    optimizer = torch.optim.Adam(dp_detector.parameters(), lr=args.lr)
    if args.optimizer_weights:
        print("Loading weights: %s" % args.optimizer_weights)
        optimizer.load_state_dict(torch.load(args.optimizer_weights))

    logger = util.Logger(args.enable_visdom, ['train_loss'])
    data_loader = DataLoader(data.ParkingSlotDataset(args.dataset_directory),
                             batch_size=args.batch_size, shuffle=True,
                             num_workers=args.data_loading_workers,
                             collate_fn=lambda x: list(zip(*x)))
    import matplotlib.pyplot as plt
    for epoch_idx in range(args.num_epochs):
        for iter_idx, (images, marking_points) in enumerate(data_loader):
            images = torch.stack(images).to(device)
            # tmp = images.cpu().numpy()[0]
            # color_image_data = np.transpose(tmp, (1, 2, 0))
            # plt.imshow(color_image_data)

            optimizer.zero_grad()
            prediction = dp_detector(images)
            objective, gradient = generate_objective(marking_points, device)

            # b, channel, w, h = objective.shape
            # objective = objective.cpu().numpy()[0]
            # objective = objective * 512 - 0.5
            # plt.figure(figsize=(10, 5))
            # for ch in range(channel):
            #     plt.subplot(1, channel, ch + 1)
            #     plt.imshow(objective[ch])  # Use 'gray' for single channel, 'viridis' for multiple channels
            #     plt.title(f'Channel {ch}')
            #     plt.grid()
            #     plt.axis('off')

            # plt.tight_layout()
            # plt.show()
            # exit()
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
            
            logger.log(epoch=epoch_idx, iter=iter_idx, train_loss=train_loss.item())
            if args.enable_visdom:
                plot_prediction(logger, images, marking_points, prediction)
        os.makedirs("weights", exist_ok=True)
        torch.save(dp_detector.state_dict(),
                   'weights/dp_detector_%d.pth' % epoch_idx)
        torch.save(optimizer.state_dict(), 'weights/optimizer.pth')


if __name__ == '__main__':
    train_detector(config.get_parser_for_training().parse_args())
