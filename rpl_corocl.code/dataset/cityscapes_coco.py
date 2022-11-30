import torch
import numpy
import random
from PIL import Image
from dataset.training.cityscapes import Cityscapes
from dataset.training.coco import COCO


class CityscapesCocoMix(torch.utils.data.Dataset):
    def __init__(self, split, preprocess, cs_root='', coco_root="", cs_split=None, coco_split=None):

        self._split_name = split
        self.preprocess = preprocess

        if cs_split is None or coco_split is None:
            self.cs_split = split
            self.coco_split = split
        else:
            self.cs_split = cs_split
            self.coco_split = coco_split
        self.city = Cityscapes(root=cs_root, split=self.cs_split)
        self.coco = COCO(root=coco_root, split=self.coco_split,
                         proxy_size=int(len(self.city)))
        self.city_number = len(self.city.images)
        self.ood_number = len(self.coco.images)
        self.train_id_out = self.coco.train_id_out
        self.num_classes = self.city.num_train_ids
        self.mean = self.city.mean
        self.std = self.city.std
        self.void_ind = self.city.ignore_in_eval_ids

    def __getitem__(self, idx):
        city_idx, anomaly_mix_or_not = idx[0], idx[1]
        # city_idx = idx
        """Return raw image, ground truth in PIL format and absolute path of raw image as string"""
        city_image = numpy.array(Image.open(self.city.images[city_idx]).convert('RGB'), dtype=numpy.float)
        city_target = numpy.array(Image.open(self.city.targets[city_idx]).convert('L'), dtype=numpy.long)

        ood_idx = random.randint(0, self.ood_number-1)
        ood_image = numpy.array(Image.open(self.coco.images[ood_idx]).convert('RGB'), dtype=numpy.float)
        ood_target = numpy.array(Image.open(self.coco.targets[ood_idx]).convert('L'), dtype=numpy.long)

        city_image, city_target, city_mix_image, city_mix_target, \
            ood_image, ood_target = self.preprocess(city_image, city_target, ood_image, ood_target,
                                                    anomaly_mix_or_not=anomaly_mix_or_not)

        return torch.tensor(city_image, dtype=torch.float), torch.tensor(city_target, dtype=torch.long), \
            torch.tensor(city_mix_image, dtype=torch.float), torch.tensor(city_mix_target, dtype=torch.long), \
            torch.tensor(ood_image, dtype=torch.float), torch.tensor(ood_target, dtype=torch.long)

    def __len__(self):
        """Return total number of images in the whole dataset."""
        return len(self.city.images)

    def __repr__(self):
        """Return number of images in each dataset."""
        fmt_str = 'Cityscapes Split: %s\n' % self.cs_split
        fmt_str += '----Number of images: %d\n' % len(self.city)
        fmt_str += 'COCO Split: %s\n' % self.coco_split
        fmt_str += '----Number of images: %d\n' % len(self.coco)
        return fmt_str.strip()
