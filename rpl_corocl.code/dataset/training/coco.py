import os
import torch
import random
from PIL import Image
from typing import Callable, Optional


class COCO(torch.utils.data.Dataset):
    train_id_in = 0
    train_id_out = 254
    min_image_size = 480

    def __init__(self, root: str, proxy_size: int, split: str = "train",
                 transform: Optional[Callable] = None, shuffle=True) -> None:
        """
        COCO dataset loader
        """
        self.root = root
        self.coco_year = '2017'
        self.split = split + self.coco_year
        self.images = []
        self.targets = []
        self.transform = transform

        for root, _, filenames in os.walk(os.path.join(self.root, "annotations", "ood_seg_" + self.split)):
            assert self.split in ['train' + self.coco_year, 'val' + self.coco_year]
            for filename in filenames:
                if os.path.splitext(filename)[-1] == '.png':
                    self.targets.append(os.path.join(root, filename))
                    self.images.append(os.path.join(self.root, self.split, filename.split(".")[0] + ".jpg"))

        """
        shuffle data and subsample
        """

        if shuffle:
            zipped = list(zip(self.images, self.targets))
            random.shuffle(zipped)
            self.images, self.targets = zip(*zipped)

        if proxy_size is not None:
            self.images = list(self.images[:int(proxy_size)])
            self.targets = list(self.targets[:int(proxy_size)])
        else:
            self.images = list(self.images[:5000])
            self.targets = list(self.targets[:5000])

    def __len__(self):
        """Return total number of images in the whole dataset."""
        return len(self.images)

    def __getitem__(self, i):
        """Return raw image and ground truth in PIL format or as torch tensor"""
        image = Image.open(self.images[i]).convert('RGB')
        target = Image.open(self.targets[i]).convert('L')
        if self.transform is not None:
            image, target = self.transform(image, target)

        return image, target

    def __repr__(self):
        """Return number of images in each dataset."""

        fmt_str = 'Number of COCO Images: %d\n' % len(self.images)
        return fmt_str.strip()
