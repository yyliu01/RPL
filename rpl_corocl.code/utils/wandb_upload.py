import os

import PIL
import matplotlib.pyplot as plt
import numpy
import torch
import torchvision
import wandb


# from utils.visualize import show_img


def get_class_colors(*args):
    # return [[128, 64, 128], [244, 35, 232], [70, 70, 70],
    #         [102, 102, 156], [190, 153, 153], [153, 153, 153],
    #         [250, 170, 30], [220, 220, 0], [107, 142, 35],
    #         [152, 251, 152], [70, 130, 180], [220, 20, 60], [255, 0, 0],
    #         [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100],
    #         [0, 0, 230], [119, 11, 32]]
    return [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153,
            153, 153, 153, 250, 170, 30,
            220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60,
            255, 0, 0, 0, 0, 142, 0, 0, 70,
            0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]


def set_img_color(colors, background, img, pred):
    for i in range(0, len(colors)):
        img[numpy.where(pred == i)] = colors[i]

    if len(pred[pred > len(colors)]) > 0:
        # original color
        if numpy.any(pred == 255):
            img[numpy.where(pred == 255)] = [0, 0, 0]
        # change to pure white; class == 19
        else:
            img[numpy.where(pred > len(colors))] = [255, 255, 255]
    return img / 255.0


class Tensorboard:
    def __init__(self, config):
        os.environ['WANDB_API_KEY'] = config['wandb_key']
        os.system("wandb login")
        os.system("wandb {}".format("online" if config['wandb_online'] else "offline"))
        self.palette = get_class_colors()
        self.tensor_board = wandb.init(project=config['proj_name'],
                                       name=config['experiment_name'],
                                       config=config)

        self.restore_transform = torchvision.transforms.Compose([
            DeNormalize(config['image_mean'], config['image_std']),
            torchvision.transforms.ToPILImage()])

    def upload_wandb_info(self, current_step, info_dict, measure_way="energy"):
        for i, info in enumerate(info_dict):
            self.tensor_board.log({info: info_dict[info], "global_step": current_step})
        return

    def upload_ood_image(self, current_step, energy_map, img_number=4, data_name="?", measure_way="energy"):
        self.tensor_board.log({"{}_focus_area_map".format(data_name): [wandb.Image(j, caption="id {}".format(str(i)))
                                                                       for i, j in enumerate(energy_map[:img_number])],
                               "global_step": current_step})

        return

    def upload_wandb_image(self, images, ground_truth, city_prediction, ood_images=None,
                           ood_prediction=None, ood_gt=None, img_number=4):

        img_number = min(ground_truth.shape[0], img_number)
        energy_map_city = -(1. * torch.logsumexp(city_prediction, dim=1))
        upload_city_ground_truth = []
        for i in range(0, img_number):
            clean_pad = numpy.zeros_like(images[0].permute(1, 2, 0).cpu().numpy())
            upload_city_ground_truth.append(set_img_color(get_class_colors(), -1, clean_pad,
                                                          ground_truth[i].detach().cpu().numpy()))

        images = self.de_normalize(images[:img_number])
        self.tensor_board.log({"city_image": [wandb.Image(j, caption="id {}".format(str(i)))
                                              for i, j in enumerate(images)]})
        self.tensor_board.log({"city_gt": [wandb.Image(j, caption="id {}".format(str(i)))
                                           for i, j in enumerate(upload_city_ground_truth)]})
        self.tensor_board.log({"city_energy_map": [wandb.Image(j, caption="id {}".format(str(i)))
                                                   for i, j in enumerate(energy_map_city)]})

        if ood_images is not None and ood_gt is not None and ood_prediction is not None:
            ood_images = self.de_normalize(ood_images[:img_number])
            energy_map_ood = -(1. * torch.logsumexp(ood_prediction, dim=1))
            self.tensor_board.log({"ood_image": [wandb.Image(j, caption="id {}".format(str(i)))
                                                 for i, j in enumerate(ood_images)]})
            self.tensor_board.log({"ood_gt": [wandb.Image(j, caption="id {}".format(str(i)))
                                              for i, j in enumerate(ood_gt.cpu().numpy())]})
            self.tensor_board.log({"ood_energy_map": [wandb.Image(j, caption="id {}".format(str(i)))
                                                      for i, j in enumerate(energy_map_ood)]})

    def de_normalize(self, image):
        return [self.restore_transform(i.detach().cpu()) if (isinstance(i, torch.Tensor) and len(i.shape) == 3)
                else colorize_mask(i.detach().cpu().numpy(), self.palette)
                for i in image]

    @staticmethod
    def finish():
        wandb.finish()


class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


def colorize_mask(mask, palette):
    zero_pad = 256 * 3 - len(palette)
    for i in range(zero_pad):
        palette.append(0)
    new_mask = PIL.Image.fromarray(mask.astype(numpy.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask
