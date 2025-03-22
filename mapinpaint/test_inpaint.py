import os
from argparse import ArgumentParser
import pandas as pd
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils

from data.dataset import Dataset
from model.networks import Generator
from utils.tools import get_config


def main():
    run_path = '../checkpoints/wgan_1616_noL1'
    checkpoint_path = f'{run_path}/gen_00400000.pt'
    config_path = f'{run_path}/config.yaml'
    test_path = '../dataset/maps_test_inpaint'
    save_img = True
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
    netG.eval()
    if cuda:
        netG = netG.cuda()
    print("Resume from {}".format(checkpoint_path))

    # Dataset
    test_dataset = Dataset(data_path=test_path,
                           image_shape=config['image_shape'],
                           data_aug=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=1,
                                              shuffle=False,
                                              num_workers=1)
    iterable_test_loader = iter(test_loader)

    results = []

    for n in range(len(test_loader)):
        ground_truth, x, mask = next(iterable_test_loader)
        if cuda:
            x = x.cuda()
            mask = mask.cuda()
            ground_truth = ground_truth.cuda()
        x_out = netG(x, mask)
        inpainted_result = x_out * mask + x * (1. - mask)

        explored_rate = ((x > 0.99).sum() / (ground_truth > 0.99).sum()).item()
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

        results.append({
            "explored_rate": explored_rate,
            "mae": mae,
            "iou": iou,
            "f1": f1
        })

        if save_img:
            viz_images = torch.stack([x, inpainted_result, ground_truth], dim=1)
            viz_images = viz_images.view(-1, *list(x.size())[1:])
            vutils.save_image(viz_images,
                              f"{run_path}/images/{n:03d}.png",
                              normalize=True)

    df = pd.DataFrame(results)
    df = df.sort_values(by="explored_rate", ascending=True)
    df.to_csv(f"{run_path}/traj_metrics.csv", index=False)


if __name__ == '__main__':
    main()
