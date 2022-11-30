from dataset.cityscapes_coco import CityscapesCocoMix
import numpy
from dataset.mix_ood_sampler import MixContextLoader
from torch.utils import data
from utils.aug import *
from utils.img_utils import generate_random_crop_pos, random_crop_pad_to_shape, random_mirror, normalize, random_scale, \
    center_crop_to_shape, pad_image_to_shape


class Preprocess(object):
    def __init__(self, config):
        self.img_mean = config.image_mean
        self.img_std = config.image_std
        self.city_image_height = config.city_image_height
        self.city_image_width = config.city_image_width
        self.ood_image_height = config.ood_image_height
        self.ood_image_width = config.ood_image_width
        self.ood_scale_array = config.ood_train_scale_array

    def inlier_transform(self, img_, gt_):
        img_, gt_ = random_mirror(img_, gt_)
        img_ = normalize(img_, self.img_mean, self.img_std)
        crop_size = (self.city_image_height, self.city_image_width)
        crop_pos = generate_random_crop_pos(img_.shape[:2], crop_size)
        img_, _ = random_crop_pad_to_shape(img_, crop_pos, crop_size, 0)
        gt_, _ = random_crop_pad_to_shape(gt_, crop_pos, crop_size, 255)
        return img_, gt_

    def outlier_transform(self, img_, gt_):
        img_ = normalize(img_, self.img_mean, self.img_std)
        scaled_img_, scaled_gt_, _ = random_scale(img_.copy(), gt_.copy(), self.ood_scale_array)
        if img_.shape[0] > self.ood_image_width or img_.shape[1] > self.ood_image_height:
            img_, gt_ = center_crop_to_shape(img=img_, gt=gt_, size=(self.ood_image_height, self.ood_image_width))
        else:
            img_, _ = pad_image_to_shape(img_, (self.ood_image_height, self.ood_image_width), value=0)
            gt_, _ = pad_image_to_shape(gt_, (self.ood_image_height, self.ood_image_width), value=255)

        if scaled_img_.shape[0] > self.ood_image_width or scaled_img_.shape[1] > self.ood_image_height:
            scaled_img_, scaled_gt_ = center_crop_to_shape(img=scaled_img_, gt=scaled_gt_, size=(self.ood_image_height, self.ood_image_width))
        else:
            scaled_img_, _ = pad_image_to_shape(scaled_img_, (self.ood_image_height, self.ood_image_width), value=0)
            scaled_gt_, _ = pad_image_to_shape(scaled_gt_, (self.ood_image_height, self.ood_image_width), value=255)

        return img_, gt_, scaled_img_, scaled_gt_

    def __call__(self, city_img, city_gt, ood_img, ood_gt, anomaly_mix_or_not):
        city_img, city_gt = self.inlier_transform(city_img, city_gt)
        ood_img, ood_gt, scaled_ood_img, scaled_ood_gt = self.outlier_transform(img_=ood_img, gt_=ood_gt)
        assert ood_img.shape == city_img.shape and ood_img.shape == scaled_ood_img.shape, \
            print("ood_img.shape {}, city_img.shape {}".format(
            ood_img.shape, city_img.shape
        ))

        assert city_gt.shape == ood_gt.shape, print("ood_img.shape {}, city_img.shape {}".format(
            ood_gt.shape, city_gt.shape
        ))

        if anomaly_mix_or_not:
            city_mix_img, city_mix_gt = self.mix_object(current_labeled_image=city_img.copy(),
                                                        current_labeled_mask=city_gt.copy(),
                                                        cut_object_image=scaled_ood_img,
                                                        cut_object_mask=scaled_ood_gt)
        else:
            city_mix_img = numpy.zeros_like(city_img)
            city_mix_gt = numpy.zeros_like(city_gt)

        return city_img.transpose(2, 0, 1), city_gt, city_mix_img.transpose(2, 0, 1), city_mix_gt, \
            ood_img.transpose(2, 0, 1), ood_gt

    def mix_object(self, current_labeled_image=None, current_labeled_mask=None,
                   cut_object_image=None, cut_object_mask=None):

        train_id_out = 254
        cut_object_mask[cut_object_mask == train_id_out] = 254

        mask = cut_object_mask == 254

        ood_mask = numpy.expand_dims(mask, axis=2)
        ood_boxes = self.extract_bboxes(ood_mask)
        ood_boxes = ood_boxes[0, :]
        y1, x1, y2, x2 = ood_boxes[0], ood_boxes[1], ood_boxes[2], ood_boxes[3]
        cut_object_mask = cut_object_mask[y1:y2, x1:x2]
        cut_object_image = cut_object_image[y1:y2, x1:x2, :]
        idx = numpy.transpose(numpy.repeat(numpy.expand_dims(cut_object_mask, axis=0), 3, axis=0), (1, 2, 0))

        h_start_point = random.randint(0, current_labeled_mask.shape[0] - cut_object_mask.shape[0])
        h_end_point = h_start_point + cut_object_mask.shape[0]
        w_start_point = random.randint(0, current_labeled_mask.shape[1] - cut_object_mask.shape[1])
        w_end_point = w_start_point + cut_object_mask.shape[1]

        current_labeled_image[h_start_point:h_end_point, w_start_point:w_end_point, :][numpy.where(idx == 254)] = \
            cut_object_image[numpy.where(idx == 254)]

        current_labeled_mask[h_start_point:h_end_point, w_start_point:w_end_point][numpy.where(cut_object_mask == 254)] = \
            cut_object_mask[numpy.where(cut_object_mask == 254)]

        return current_labeled_image, current_labeled_mask

    @staticmethod
    def extract_bboxes(mask):
        boxes = numpy.zeros([mask.shape[-1], 4], dtype=numpy.int32)
        for i in range(mask.shape[-1]):
            m = mask[:, :, i]
            # Bounding box.
            horizontal_indicies = numpy.where(numpy.any(m, axis=0))[0]
            vertical_indicies = numpy.where(numpy.any(m, axis=1))[0]
            if horizontal_indicies.shape[0]:
                x1, x2 = horizontal_indicies[[0, -1]]
                y1, y2 = vertical_indicies[[0, -1]]
                # x2 and y2 should not be part of the box. Increment by 1.
                x2 += 1
                y2 += 1
            else:
                # No mask for this instance. Might happen due to
                # resizing or cropping. Set bbox to zeros
                x1, x2, y1, y2 = 0, 0, 0, 0
            boxes[i] = numpy.array([y1, x1, y2, x2])
        return boxes.astype(numpy.int32)


def get_mix_loader(engine, config, proba_factor=1.):
    train_dataset = CityscapesCocoMix(split='train', preprocess=Preprocess(config),
                                      cs_root=config.city_root_path, coco_root=config.coco_root_path)

    train_sampler = None
    is_shuffle = True

    if engine.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        is_shuffle = False

    train_loader = MixContextLoader(dataset=train_dataset, shuffle=is_shuffle, num_workers=config.num_workers,
                                    batch_size=config.batch_size, city_img_num=config.num_train_imgs,
                                    coco_img_num=int(proba_factor*config.num_train_imgs), gpu_num=engine.gpus)

    return train_loader, train_sampler



