import os
from argparse import ArgumentParser
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils

from data.dataset import Dataset
from model.networks import Generator, Discriminator
from utils.tools import get_config


class Evaluator:
    def __init__(self, config, netG=None, nsample=0):
        self.config = config
        self.use_cuda = self.config['cuda']
        self.device_ids = self.config['gpu_ids']
        self.nsample = nsample
        if netG is not None:
            self.netG = netG
        else:
            self.netG = Generator(self.config['netG'], self.use_cuda, self.device_ids)
        if self.use_cuda:
            self.netG.to(self.device_ids[0])

    @torch.no_grad()
    def eval_step(self, x, mask, ground_truth, onehot, img_raw_size):
        self.netG.eval()
        x_out, _, _ = self.netG(x, mask, onehot)
        inpainted_result = x_out * mask + x * (1. - mask)

        width, height = x.size(2), x.size(3)
        if img_raw_size[0] < width and img_raw_size[1] < height:
            i_left = (width - img_raw_size[0]) // 2
            i_top = (height - img_raw_size[1]) // 2
            i_right = i_left + img_raw_size[0]
            i_bottom = i_top + img_raw_size[1]
            x = x[:, :, i_left:i_right, i_top:i_bottom]
            ground_truth = ground_truth[:, :, i_left:i_right, i_top:i_bottom]
            inpainted_result = inpainted_result[:, :, i_left:i_right, i_top:i_bottom]
        else:
            # reshape
            x = F.interpolate(x, size=(img_raw_size[1], img_raw_size[0]), mode='bilinear', align_corners=False)
            ground_truth = F.interpolate(ground_truth, size=(img_raw_size[1], img_raw_size[0]), mode='bilinear', align_corners=False)
            inpainted_result = F.interpolate(inpainted_result, size=(img_raw_size[1], img_raw_size[0]), mode='bilinear', align_corners=False)

        mae = F.l1_loss(inpainted_result, ground_truth).item()
        pred_mask_flat = (inpainted_result > 0).view(-1)
        gt_mask_flat = (ground_truth > 0).view(-1)
        intersection = (pred_mask_flat & gt_mask_flat).sum().float()
        union = (pred_mask_flat | gt_mask_flat).sum().float()
        TP = intersection
        FP = (pred_mask_flat & ~gt_mask_flat).sum().float()
        FN = (~pred_mask_flat & gt_mask_flat).sum().float()
        iou = (intersection / union).item()
        f1 = (2 * TP / (2 * TP + FP + FN)).item()

        metrics = {'mae': mae, 'iou': iou, 'f1': f1}

        return metrics, inpainted_result


def post_process(inpaint, x, kernel_size=5):
    unique_values, counts = torch.unique(x, return_counts=True)
    top3_indices = torch.topk(counts, k=3).indices
    top3_values = unique_values[top3_indices]
    obs_v, free_v = top3_values.min(), top3_values.max()

    inpaint = torch.where(inpaint > -0.3, free_v, obs_v)  # binarization
    binary_img = inpaint.cpu().numpy()[0, 0]
    obs_v = obs_v.item()
    free_v = free_v.item()

    mask = np.zeros_like(binary_img, dtype=np.uint8)
    mask[binary_img == free_v] = 255
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    opening = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # close op to fill small holes
    opening = cv2.morphologyEx(opening, cv2.MORPH_OPEN, kernel)  # open op to remove small noise
    morph_clean_img = np.where(opening == 255, free_v, obs_v).astype(binary_img.dtype)
    x_array = x.cpu().numpy()[0, 0]
    morph_clean_img = np.where((x_array == obs_v) | (x_array == free_v), x_array, morph_clean_img)
    return morph_clean_img


def main():
    run_path = '../checkpoints/wgan_'
    config_path = f'{run_path}/config.yaml'
    checkpoint_path = os.path.join(run_path, [f for f in os.listdir(run_path) if f.startswith('gen') and f.endswith('.pt')][0])
    save_img = True
    save_csv = True
    nsample = 5  # set 1, >=2 to set the number of samples
    if save_img:
        os.makedirs(f"{run_path}/images", exist_ok=True)
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default=config_path, help="testing configuration")
    parser.add_argument('--seed', type=int, default=0, help='manual seed')

    args = parser.parse_args()
    config = get_config(args.config)
    print("Arguments: {}".format(args))

    # CUDA configuration
    cuda = config['cuda']
    device_ids = config['gpu_ids']
    if cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in device_ids)
        device_ids = list(range(len(device_ids)))
        config['gpu_ids'] = device_ids
        cudnn.benchmark = True
    print("Configuration: {}".format(config))

    # Define the trainer
    netG = Generator(config['netG'], cuda, device_ids)
    netG.load_state_dict(torch.load(checkpoint_path))
    print("Resume from {}".format(checkpoint_path))

    evaluator = Evaluator(config, netG, nsample)

    # Dataset
    # config['eval_data_path'] = '../dataset/kth_test_maps/50052748'
    eval_dataset = Dataset(data_path=config['eval_data_path'],
                           image_shape=config['image_shape'],
                           data_aug=False)
    eval_loader = torch.utils.data.DataLoader(dataset=eval_dataset,
                                              batch_size=1,
                                              shuffle=False,
                                              num_workers=1)
    iterable_eval_loader = iter(eval_loader)

    results = []

    for n in range(len(eval_loader)):
        ground_truth, x, mask, _, (ground_truth_raw, x_raw, mask_raw) = next(iterable_eval_loader)
        if cuda:
            x = x.cuda()
            mask = mask.cuda()
            ground_truth = ground_truth.cuda()
            x_raw = x_raw.cuda()
            ground_truth_raw = ground_truth_raw.cuda()
        explored_rate = ((x > 0.99).sum() / (ground_truth > 0.99).sum()).item()

        all_inpaints = []
        all_inpaints_processed = []
        onehots = torch.tensor([[1,0],[0,1],[0.5,0.5],[0,0],[1,1]]).unsqueeze(1).float().to(x.device)

        for i in range(nsample):
            metrics, inpainted_result = evaluator.eval_step(x, mask, ground_truth, onehots[i], eval_dataset.image_raw_shape)

            inpaint_processed = post_process(inpainted_result, x_raw)
            inpaint_processed = torch.from_numpy(inpaint_processed).unsqueeze(0).unsqueeze(0).float().to(x.device)
            all_inpaints.append(inpainted_result)
            all_inpaints_processed.append(inpaint_processed)

            log_metrics = {"explored_rate": explored_rate,
                           "mae": metrics['mae'],
                           "iou": metrics['iou'],
                           "f1": metrics['f1']}

            results.append(log_metrics)

        if save_img:
            avg_inpainted_result = torch.stack(all_inpaints, dim=0).mean(dim=0)
            avg_inpainted_processed = torch.stack(all_inpaints_processed, dim=0).mean(dim=0)
            viz_images = torch.stack([x_raw, avg_inpainted_result, avg_inpainted_processed, ground_truth_raw], dim=1)
            viz_images = viz_images.view(-1, *list(x_raw.size())[1:])
            vutils.save_image(viz_images,
                              f"{run_path}/images/{n:03d}.png",
                              normalize=True)
            if nsample > 1:
                all_inpaints = torch.cat(all_inpaints, dim=0)
                vutils.save_image(all_inpaints,
                                  f"{run_path}/images/{n:03d}_sample.png",
                                  normalize=True)

    if save_csv:
        df = pd.DataFrame(results)
        df = df.sort_values(by="explored_rate", ascending=True)
        df.to_csv(f"{run_path}/traj_metrics.csv", index=False)
    print(f"Mean mae: {np.mean([y['mae'] for y in results]):.4f}")
    print(f"Mean iou: {np.mean([y['iou'] for y in results]):.4f}")
    print(f"Mean f1: {np.mean([y['f1'] for y in results]):.4f}")


if __name__ == '__main__':
    main()
