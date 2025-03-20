import sys
import random
import numpy as np
import torch.utils.data as data
from os import listdir
from PIL import Image
from attr.filters import exclude

from utils.tools import default_loader, ground_truth_loader, is_image_file, normalize
import os

import torchvision.transforms as transforms


class Dataset(data.Dataset):
    def __init__(self, data_path, image_shape):
        super(Dataset, self).__init__()
        self.ground_truth = [x for x in listdir(f"{data_path}/full") if is_image_file(x)]
        self.partial_map = [x for x in listdir(f"{data_path}/part") if is_image_file(x)]
        self.map_mask = [x for x in listdir(f"{data_path}/mask") if is_image_file(x)]
        self.data_path = data_path
        self.image_shape = image_shape[:-1]

    @staticmethod
    def rotate_img(img, rot):
        rotations = {
            0: lambda x: x,
            90: lambda x: x.transpose(Image.ROTATE_90),
            180: lambda x: x.transpose(Image.ROTATE_180),
            270: lambda x: x.transpose(Image.ROTATE_270)
        }
        return rotations[rot](img)

    def crop_img(self, path, img_type=None, rotation=0):
        if img_type == 'mask':
            img = default_loader(path)
            img = img.convert('1')
        elif img_type == 'belief':
            img = default_loader(path)
            img = img.convert('L')
        elif img_type == 'gt':
            img = ground_truth_loader(path)
            img = img.convert('L')
        else:
            raise ValueError

        img = self.rotate_img(img, rotation)

        width, height = img.size
        if width < self.image_shape[0] and height < self.image_shape[1]:
            pad_left = (self.image_shape[0] - width) // 2
            pad_top = (self.image_shape[1] - height) // 2
            pad_right = self.image_shape[0] - width - pad_left
            pad_bottom = self.image_shape[1] - height - pad_top
            img = transforms.Pad((pad_left, pad_top, pad_right, pad_bottom), padding_mode='edge')(img)
        else:
            img = transforms.Resize(self.image_shape)(img)
        # img = transforms.RandomCrop(self.image_shape)(img)
        img = transforms.ToTensor()(img)  # turn the image to a tensor
        if img_type != 'mask':
            img = normalize(img)
        return img

    def __getitem__(self, index):
        data_aug = random.choice([0, 90, 180, 270])
        partial_img = self.crop_img(os.path.join(f"{self.data_path}/part", self.partial_map[index]), img_type='belief', rotation=data_aug)
        mask_img = self.crop_img(os.path.join(f"{self.data_path}/mask", self.map_mask[index]), img_type='mask', rotation=data_aug)
        map_name = '_'.join(self.partial_map[index].split('_')[:-1])
        ground_truth = self.crop_img(os.path.join(f"{self.data_path}/full", f"{map_name}.png"), img_type='gt', rotation=data_aug)
        return ground_truth, partial_img, mask_img

    def __len__(self):
        return len(self.partial_map)
