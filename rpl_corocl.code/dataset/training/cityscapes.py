import os

import numpy as np
import torch
from PIL import Image
from collections import namedtuple
from typing import Any, Callable, Optional, Tuple


class Cityscapes(torch.utils.data.Dataset):
    """`
    Cityscapes Dataset http://www.cityscapes-dataset.com/
    Labels based on https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
    """
    CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                     'has_instances', 'ignore_in_eval', 'color'])

    labels = [
        CityscapesClass('unlabeled', 0, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('ego vehicle', 1, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('out of roi', 3, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('static', 4, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('dynamic', 5, 255, 'void', 0, False, True, (111, 74, 0)),
        CityscapesClass('ground', 6, 255, 'void', 0, False, True, (81, 0, 81)),
        CityscapesClass('road', 7, 0, 'flat', 1, False, False, (128, 64, 128)),
        CityscapesClass('sidewalk', 8, 1, 'flat', 1, False, False, (244, 35, 232)),
        CityscapesClass('parking', 9, 255, 'flat', 1, False, True, (250, 170, 160)),
        CityscapesClass('rail track', 10, 255, 'flat', 1, False, True, (230, 150, 140)),
        CityscapesClass('building', 11, 2, 'construction', 2, False, False, (70, 70, 70)),
        CityscapesClass('wall', 12, 3, 'construction', 2, False, False, (102, 102, 156)),
        CityscapesClass('fence', 13, 4, 'construction', 2, False, False, (190, 153, 153)),
        CityscapesClass('guard rail', 14, 255, 'construction', 2, False, True, (180, 165, 180)),
        CityscapesClass('bridge', 15, 255, 'construction', 2, False, True, (150, 100, 100)),
        CityscapesClass('tunnel', 16, 255, 'construction', 2, False, True, (150, 120, 90)),
        CityscapesClass('pole', 17, 5, 'object', 3, False, False, (153, 153, 153)),
        CityscapesClass('polegroup', 18, 255, 'object', 3, False, True, (153, 153, 153)),
        CityscapesClass('traffic light', 19, 6, 'object', 3, False, False, (250, 170, 30)),
        CityscapesClass('traffic sign', 20, 7, 'object', 3, False, False, (220, 220, 0)),
        CityscapesClass('vegetation', 21, 8, 'nature', 4, False, False, (107, 142, 35)),
        CityscapesClass('terrain', 22, 9, 'nature', 4, False, False, (152, 251, 152)),
        CityscapesClass('sky', 23, 10, 'sky', 5, False, False, (70, 130, 180)),
        CityscapesClass('person', 24, 11, 'human', 6, True, False, (220, 20, 60)),
        CityscapesClass('rider', 25, 12, 'human', 6, True, False, (255, 0, 0)),
        CityscapesClass('car', 26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
        CityscapesClass('truck', 27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
        CityscapesClass('bus', 28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
        CityscapesClass('caravan', 29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
        CityscapesClass('trailer', 30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
        CityscapesClass('train', 31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
        CityscapesClass('motorcycle', 32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
        CityscapesClass('bicycle', 33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
        CityscapesClass('license plate', -1, -1, 'vehicle', 7, False, True, (0, 0, 142)),
    ]

    """Normalization parameters"""
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    """Useful information from labels"""
    ignore_in_eval_ids, label_ids, train_ids, train_id2id = [], [], [], []  # empty lists for storing ids
    color_palette_train_ids = [(0, 0, 0) for i in range(256)]
    for i in range(len(labels)):
        if labels[i].ignore_in_eval and labels[i].train_id not in ignore_in_eval_ids:
            ignore_in_eval_ids.append(labels[i].train_id)
    for i in range(len(labels)):
        label_ids.append(labels[i].id)
        if labels[i].train_id not in ignore_in_eval_ids:
            train_ids.append(labels[i].train_id)
            color_palette_train_ids[labels[i].train_id] = labels[i].color
            train_id2id.append(labels[i].id)
    num_label_ids = len(set(label_ids))  # Number of ids
    num_train_ids = len(set(train_ids))  # Number of trainIds
    id2label = {label.id: label for label in labels}
    train_id2label = {label.train_id: label for label in labels}

    def __init__(self, root: str = "/path/to/you/root", split: str = "val", mode: str = "gtFine",
                 target_type: str = "semantic_train_id", transform: Optional[Callable] = None,
                 predictions_root: Optional[str] = None) -> None:
        """
        Cityscapes dataset loader
        """
        self.root = root
        self.split = split
        self.mode = 'gtFine' if "fine" in mode.lower() else 'gtCoarse'
        self.transform = transform
        self.images_dir = os.path.join(self.root, 'images', 'city_gt_fine', self.split)
        self.targets_dir = os.path.join(self.root, 'annotation', 'city_gt_fine', self.split)
        self.predictions_dir = os.path.join(predictions_root, self.split) if predictions_root is not None else ""
        self.target_type = target_type

        self.images = []
        self.targets = []
        self.predictions = []

        img_dir = self.images_dir
        target_dir = self.targets_dir
        pred_dir = self.predictions_dir
        for file_name in os.listdir(img_dir):
            target_name = '{}_{}'.format(file_name.split('_leftImg8bit')[0],
                                         self._get_target_suffix(self.mode, target_type))
            self.images.append(os.path.join(img_dir, file_name))
            self.targets.append(os.path.join(target_dir, target_name))
            self.predictions.append(os.path.join(pred_dir, file_name.replace("_leftImg8bit", "")))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image = Image.open(self.images[index]).convert('RGB')

        if self.split in ['train', 'val']:
            target = Image.open(self.targets[index])
            if self.transform is not None:
                image, target = self.transform(image, target)
            return image, target
        else:
            target = Image.fromarray(np.zeros([image.size[1], image.size[0]]))
            if self.transform is not None:
                image, target = self.transform(image, target)
            return image, '{}_{}'.format(self.images[index].split('/')[-1].split('_leftImg8bit')[0],
                                         self._get_target_suffix(self.mode, self.target_type))

    def __len__(self) -> int:
        return len(self.images)

    @staticmethod
    def _get_target_suffix(mode: str, target_type: str) -> str:
        if target_type == 'instance':
            return '{}_instanceIds.png'.format(mode)
        elif target_type == 'semantic_id':
            return '{}_labelIds.png'.format(mode)
        elif target_type == 'semantic_train_id':
            return '{}.png'.format(mode)
        elif target_type == 'color':
            return '{}_color.png'.format(mode)
        else:
            print("'%s' is not a valid target type, choose from:\n" % target_type +
                  "['instance', 'semantic_id', 'semantic_train_id', 'color']")
            exit()
