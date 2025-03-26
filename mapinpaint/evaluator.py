import os
from argparse import ArgumentParser
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils

from data.dataset import Dataset
from model.networks import Generator, Discriminator
from utils.tools import get_config


class Evaluator:
    def __init__(self, config, netG=None):
        self.config = config
        self.use_cuda = self.config['cuda']
        self.device_ids = self.config['gpu_ids']
        if netG is not None:
            self.netG = netG
        else:
            self.netG = Generator(self.config['netG'], self.use_cuda, self.device_ids)
        if self.use_cuda:
            self.netG.to(self.device_ids[0])

    @torch.no_grad()
    def eval_step(self, x, mask, ground_truth, img_raw_size):
        self.netG.eval()
        x_out = self.netG(x, mask)
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

        return metrics, (inpainted_result, x, ground_truth)


def post_process(inpaint, obs_v, free_v, kernel_size=9):
    inpaint = torch.where(inpaint > -0.3, free_v, obs_v)
    binary_img = inpaint.cpu().numpy()[0, 0]
    obs_v = obs_v.item()
    free_v = free_v.item()

    mask = np.zeros_like(binary_img, dtype=np.uint8)
    mask[binary_img == free_v] = 255
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    opening = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # close op to fill small holes
    opening = cv2.morphologyEx(opening, cv2.MORPH_OPEN, kernel)  # open op to remove small noise
    morph_clean_img = np.where(opening == 255, free_v, obs_v).astype(binary_img.dtype)

    return morph_clean_img


def main():
    run_path = '../checkpoints/wgan_sdnorm0.95'
    config_path = f'{run_path}/config.yaml'
    checkpoint_path = os.path.join(run_path, [f for f in os.listdir(run_path) if f.startswith('gen') and f.endswith('.pt')][0])
    save_img = True
    logD = True
    if save_img:
        if not os.path.exists(f"{run_path}/images"):
            os.makedirs(f"{run_path}/images")
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

    if logD:
        netD = Discriminator({'input_dim':1, 'ndf':32}, cuda, device_ids)
        netD.load_state_dict(torch.load('../checkpoints/wgan_sdnorm0.95_3232/dis_00340000.pt'))
        if cuda:
            netD.to(device_ids[0])

    evaluator = Evaluator(config, netG)

    # Dataset
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
        ground_truth, x, mask = next(iterable_eval_loader)
        if cuda:
            x = x.cuda()
            mask = mask.cuda()
            ground_truth = ground_truth.cuda()
        explored_rate = ((x > 0.99).sum() / (ground_truth > 0.99).sum()).item()

        metrics, (inpainted_result, x, ground_truth) = evaluator.eval_step(x, mask, ground_truth, eval_dataset.image_raw_shape)
        obs_v, free_v = torch.unique(ground_truth)

        inpaint_processed = post_process(inpainted_result, obs_v, free_v)
        inpaint_processed = torch.from_numpy(inpaint_processed).unsqueeze(0).unsqueeze(0).float().to(x.device)

        log_metrics = {"explored_rate": explored_rate,
                       "mae": metrics['mae'],
                       "iou": metrics['iou'],
                       "f1": metrics['f1']}

        if logD:
            inpaint_score = netD(inpainted_result)
            gt_score = netD(ground_truth)
            score_diff = inpaint_score - gt_score
            log_metrics.update({"inpaint_score": inpaint_score.item(),
                                "dis_score": score_diff.item()})

        results.append(log_metrics)

        if save_img:
            viz_images = torch.stack([x, inpainted_result, inpaint_processed, ground_truth], dim=1)
            viz_images = viz_images.view(-1, *list(x.size())[1:])
            vutils.save_image(viz_images,
                              f"{run_path}/images/{n:03d}.png",
                              normalize=True)

    df = pd.DataFrame(results)
    df = df.sort_values(by="explored_rate", ascending=True)
    df.to_csv(f"{run_path}/traj_metrics.csv", index=False)


if __name__ == '__main__':
    main()
