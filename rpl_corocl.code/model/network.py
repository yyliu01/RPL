import torch.nn as nn
from model.wide_network import DeepWV3Plus


class Network(nn.Module):
    def __init__(self, num_classes):
        super(Network, self).__init__()
        # self.branch1 = _segm_resnet('deeplabv3plus', "resnet101",
        #                             num_classes, 16, False)
        # self.branch1 = _segm_mobilenet('deeplabv3plus', "nil",
        #                                num_classes, output_stride=16, pretrained_backbone=False)
        self.branch1 = DeepWV3Plus(num_classes)

    def forward(self, data):
        return self.branch1(data)
