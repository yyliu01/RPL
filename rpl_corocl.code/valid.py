import os
import h5py
import matplotlib.pyplot as plt
import numpy
import torch.distributed as dist
import torch.optim
import torchvision
from PIL import Image
from tqdm import tqdm
from dataset.training.cityscapes import Cityscapes
from dataset.validation.fishyscapes import Fishyscapes
from dataset.validation.road_anomaly import RoadAnomaly
from dataset.validation.segment_me_if_you_can import SegmentMeIfYouCan
from utils.img_utils import Compose, ToTensor, Normalize
from utils.metric import eval_metrics
from utils.pyt_utils import eval_ood_measure
from utils.smiyc_metrics import SmiycMeasures
# from engine.evaluator import SlidingEval
# from dataset.validation.lost_and_found import LostAndFound


class Validator(torch.nn.Module):
    def __init__(self, config, logger):
        super(Validator, self).__init__()
        self.logger = logger
        self.eval_iter = config.eval_iter
        self.measure_way = config.measure_way
        testing_transform = Compose([ToTensor(), Normalize(config.image_mean, config.image_std)])
        self.fishyscapes_ls = Fishyscapes(split='LostAndFound', root=config.fishy_root_path,
                                          transform=testing_transform)
        self.fishyscapes_static = Fishyscapes(split='Static', root=config.fishy_root_path, transform=testing_transform)
        self.cityscapes = Cityscapes(root=config.city_root_path, split="val", transform=testing_transform)
        self.segment_me_anomaly = SegmentMeIfYouCan(split='road_anomaly', root=config.segment_me_root_path,
                                                    transform=testing_transform)
        self.segment_me_obstacle = SegmentMeIfYouCan(split='road_obstacle', root=config.segment_me_root_path,
                                                     transform=testing_transform)
        self.road_anomaly = RoadAnomaly(root=config.road_anomaly_root_path, transform=testing_transform)
        # self.lost_and_found = LostAndFound(root=config.lost_and_found_root_path, transform=testing_transform)
        # self.sm_lost_and_found = LostAndFound(root=config.segment_me_lf_root_path, transform=testing_transform)
        # self.evaluator = SlidingEval(config, device=0)

    def run(self, model, engine, curr_iter, vis_tool):
        valid_anomaly(model=model, engine=engine, iteration=curr_iter, test_set=self.segment_me_anomaly,
                      data_name='segment_me_anomaly', my_wandb=vis_tool, logger=self.logger,
                      measure_way=self.measure_way)

        valid_anomaly(model=model, engine=engine, iteration=curr_iter, test_set=self.segment_me_obstacle,
                      data_name='segment_me_obstacle', my_wandb=vis_tool, logger=self.logger,
                      measure_way=self.measure_way)

        valid_anomaly(model=model, engine=engine, iteration=curr_iter, test_set=self.fishyscapes_static,
                      data_name='Fishyscapes_static', my_wandb=vis_tool, logger=self.logger,
                      measure_way=self.measure_way)

        valid_anomaly(model=model, engine=engine, iteration=curr_iter, test_set=self.fishyscapes_ls,
                      data_name='Fishyscapes_ls', my_wandb=vis_tool, logger=self.logger,
                      measure_way=self.measure_way)

        valid_anomaly(model=model, engine=engine, iteration=curr_iter, test_set=self.road_anomaly,
                      data_name='road_anomaly',
                      my_wandb=vis_tool, logger=self.logger, measure_way=self.measure_way)


def compute_anomaly_score(score, mode='energy'):
    score = score.squeeze()[:19]
    if mode == 'energy':
        anomaly_score = -(1. * torch.logsumexp(score, dim=0))
    elif mode == 'entropy':
        prob = torch.softmax(score, dim=0)
        anomaly_score = -torch.sum(prob * torch.log(prob), dim=0) / torch.log(torch.tensor(19.))
    else:
        raise NotImplementedError

    # regular gaussian smoothing
    anomaly_score = anomaly_score.unsqueeze(0)
    anomaly_score = torchvision.transforms.GaussianBlur(7, sigma=1)(anomaly_score)
    anomaly_score = anomaly_score.squeeze(0)
    return anomaly_score


def valid_anomaly_sample_wise(model, engine, test_set, data_name, iteration, my_wandb, logger=None,
                              measure_way="energy"):
    if engine.local_rank <= 0:
        logger.info("validating {} dataset with {} ...".format(data_name, measure_way))

    curr_rank = max(0, engine.local_rank)
    stride = int(numpy.ceil(len(test_set) / engine.gpus))
    e_record = min((curr_rank + 1) * stride, len(test_set))
    shred_list = list(range(curr_rank * stride, e_record))
    curr_info = {}
    tbar = tqdm(shred_list, ncols=137, leave=True, miniters=1) if curr_rank <= 0 else shred_list
    model.eval()
    roc_aucs = []
    prc_aucs = []
    fprs = []

    with torch.no_grad():
        for idx in tbar:
            img, label, _ = test_set[idx]
            img = img.cuda(non_blocking=True)
            _, logits, _ = model.module(img) if engine.distributed else model(img)
            anomaly_score = compute_anomaly_score(logits, mode=measure_way).cpu()
            assert ~torch.isnan(anomaly_score).any(), "expecting no nan in score {}.".format(measure_way)
            roc_auc, prc_auc, fpr = eval_ood_measure(anomaly_score, label.cpu(), test_set.train_id_in,
                                                     test_set.train_id_out)
            if roc_auc is None and prc_auc is None and fpr is None:
                continue

            roc_aucs.append(roc_auc)
            prc_aucs.append(prc_auc)
            fprs.append(fpr)

            torch.cuda.empty_cache()
            del img, logits

    # evaluation
    if engine.gpus > 1:
        local_fprs = torch.tensor(fprs, device="cpu")
        global_fprs = [_ for _ in range(engine.gpus)]
        torch.distributed.all_gather_object(global_fprs, local_fprs)
        fprs = torch.cat(global_fprs).numpy()

        local_roc_aucs = torch.tensor(roc_aucs, device="cpu")
        global_roc_aucs = [_ for _ in range(engine.gpus)]
        torch.distributed.all_gather_object(global_roc_aucs, local_roc_aucs)
        roc_aucs = torch.cat(global_roc_aucs).numpy()

        local_prc_aucs = torch.tensor(prc_aucs, device="cpu")
        global_prc_aucs = [_ for _ in range(engine.gpus)]
        torch.distributed.all_gather_object(global_prc_aucs, local_prc_aucs)
        prc_aucs = torch.cat(global_prc_aucs).numpy()

    fprs = numpy.mean(fprs)
    roc_aucs = numpy.mean(roc_aucs)
    prc_aucs = numpy.mean(prc_aucs)

    curr_info['{}_fpr95'.format(data_name)] = fprs
    curr_info['{}_auroc'.format(data_name)] = roc_aucs
    curr_info['{}_auprc'.format(data_name)] = prc_aucs

    if engine.local_rank <= 0:
        logger.critical("AUROC score for {}: {:.4f}".format(data_name, roc_aucs))
        logger.critical("AUPRC score for {}: {:.4f}".format(data_name, prc_aucs))
        logger.critical("FPR@TPR95 for {}: {:.4f}".format(data_name, fprs))
        if my_wandb is not None:
            my_wandb.upload_wandb_info(current_step=iteration, info_dict=curr_info, measure_way=measure_way)

    if engine.distributed:
        dist.barrier()

    del curr_info
    return


def valid_anomaly_from_saved_results(model, engine, test_set, data_name=None, iteration=None, my_wandb=None,
                                     logger=None, upload_img_num=4, measure_way="energy"):
    saved_dir = os.path.join('.', 'rpl_corocl.code', 'lost_and_found_outputs')
    if not os.path.exists(saved_dir):
        os.mkdir(saved_dir)

    if engine.local_rank <= 0:
        logger.info("validating {} dataset with {} ...".format(data_name, measure_way))
    curr_rank = max(0, engine.local_rank)
    stride = int(numpy.ceil(len(test_set) / engine.gpus))
    e_record = min((curr_rank + 1) * stride, len(test_set))
    shred_list = list(range(curr_rank * stride, e_record))
    tbar1 = tqdm(shred_list, ncols=137, leave=True, miniters=1, desc="saving the predictions ...") if curr_rank <= 0 else shred_list
    tbar2 = tqdm(list(range(len(test_set))), ncols=137, leave=True, miniters=1, desc="calculating results ...") if curr_rank <= 0 \
        else list(range(len(test_set)))

    model.eval()
    smiyc_measures = SmiycMeasures()
    temp_list = []
    with torch.no_grad():
        for idx in tbar1:
            img, label, img_id = test_set[idx]
            img = img.cuda(non_blocking=True)
            _, logits, _ = model.module(img) if engine.distributed else model(img)
            anomaly_score = compute_anomaly_score(logits, mode=measure_way).cpu().numpy().astype(numpy.float16)
            curr_path = os.path.join(saved_dir, img_id + ".hdf5")
            with h5py.File(curr_path, 'w') as hdf5_write_handle:
                hdf5_write_handle.create_dataset('value', data=anomaly_score)
            # tbar1.set_description("saving the predictions ...")

    if engine.local_rank <= 0:
        logger.critical("validating the saved predictions ...")
        for idx in tbar2:
            _, label, img_id = test_set[idx]
            if test_set.train_id_out not in label: continue
            curr_path = os.path.join(saved_dir, img_id + ".hdf5")
            with h5py.File(curr_path, 'r') as hdf5_file_handle:
                anomaly_score = hdf5_file_handle['value'][:]
            temp_list.append(smiyc_measures(anomaly_score, label.numpy()))

        result = smiyc_measures.aggregate(temp_list)
        curr_info = {}
        fpr, prc_auc, roc_auc = result['tpr95_fpr'], result['area_PRC'], result['area_ROC']
        curr_info['{}_auroc'.format(data_name)] = roc_auc
        curr_info['{}_fpr95'.format(data_name)] = fpr
        curr_info['{}_auprc'.format(data_name)] = prc_auc

        logger.critical("AUROC score for {}: {:.4f}".format(data_name, roc_auc))
        logger.critical("AUPRC score for {}: {:.4f}".format(data_name, prc_auc))
        logger.critical("FPR@TPR95 for {}: {:.4f}".format(data_name, fpr))
        if my_wandb is not None:
            my_wandb.upload_wandb_info(current_step=iteration, info_dict=curr_info, measure_way=measure_way)

        del curr_info

    if engine.distributed:
        dist.barrier()

    return

import PIL
def get_class_colors(*args):
    return [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153,
            153, 153, 153, 250, 170, 30,
            220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60,
            255, 0, 0, 0, 0, 142, 0, 0, 70,
            0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]


class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


def de_normalize(image, instance="img"):
    palette = get_class_colors()
    restore_transform = torchvision.transforms.Compose([
        DeNormalize(numpy.array([0.485, 0.456, 0.406]), numpy.array([0.229, 0.224, 0.225])),
        torchvision.transforms.ToPILImage()])
    if instance == "img":
        return [restore_transform(i.detach().cpu()) for i in image]
    else:
        return [colorize_mask(i.detach().cpu().numpy(), palette)
                for i in image]


def colorize_mask(mask, palette):
    zero_pad = 256 * 3 - len(palette)
    for i in range(zero_pad):
        palette.append(0)
    # palette[-6:-3] = [183, 65, 14]
    new_mask = PIL.Image.fromarray(mask.astype(numpy.uint8)).convert('P')
    new_mask.putpalette(palette)
    new_mask = new_mask.convert('RGB')

    return new_mask


def NormalizeData(data):
    return (data - numpy.min(data)) / (numpy.max(data) - numpy.min(data))


def valid_anomaly(model, engine, test_set, data_name=None, iteration=None, my_wandb=None, logger=None,
                  upload_img_num=4, measure_way="energy"):
    if engine.local_rank <= 0:
        logger.info("validating {} dataset with {} ...".format(data_name, measure_way))
    # dir_name = "ours"
    # if not os.path.exists(dir_name):
    #     os.mkdir(dir_name)
    # path_name = os.path.join(dir_name, data_name)
    # if not os.path.exists(path_name):
    #     os.mkdir(path_name)

    curr_rank = max(0, engine.local_rank)
    stride = int(numpy.ceil(len(test_set) / engine.gpus))
    e_record = min((curr_rank + 1) * stride, len(test_set))
    shred_list = list(range(curr_rank * stride, e_record))
    curr_info = {}
    tbar = tqdm(shred_list, ncols=137, leave=True, miniters=1) if curr_rank <= 0 else shred_list
    model.eval()
    focus_area = []
    ood_gts_list = []
    anomaly_score_list = []
    with torch.no_grad():
        for idx in tbar:
            img, label = test_set[idx]
            img = img.cuda(non_blocking=True)
            inlier_logits, logits, _ = model.module(img) if engine.distributed else model(img)
            anomaly_score = compute_anomaly_score(logits, mode=measure_way).cpu()
            visual = anomaly_score.detach().cpu()
            assert ~torch.isnan(anomaly_score).any(), "expecting no nan in score {}.".format(measure_way)
            ood_gts_list.append(numpy.expand_dims(label.detach().numpy(), 0))
            anomaly_score_list.append(numpy.expand_dims(anomaly_score.numpy(), 0))
            visual[(label != test_set.train_id_out) & (label != test_set.train_id_in)] = 0
            focus_area.append(visual)
            torch.cuda.empty_cache()
            del img, logits

    # evaluation
    ood_gts = numpy.array(ood_gts_list)
    anomaly_scores = numpy.array(anomaly_score_list)

    if engine.gpus > 1:
        anomaly_scores, ood_gts = all_gather_samples(x_=anomaly_scores, y_=ood_gts, local_rank=engine.local_rank)

    if engine.local_rank <= 0:
        roc_auc, prc_auc, fpr = eval_ood_measure(anomaly_scores, ood_gts, test_set.train_id_in, test_set.train_id_out)

        curr_info['{}_auroc'.format(data_name)] = roc_auc
        curr_info['{}_fpr95'.format(data_name)] = fpr
        curr_info['{}_auprc'.format(data_name)] = prc_auc

        logger.critical("AUROC score for {}: {:.4f}".format(data_name, roc_auc))
        logger.critical("AUPRC score for {}: {:.4f}".format(data_name, prc_auc))
        logger.critical("FPR@TPR95 for {}: {:.4f}".format(data_name, fpr))
        if my_wandb is not None:
            my_wandb.upload_wandb_info(current_step=iteration, info_dict=curr_info, measure_way=measure_way)
            my_wandb.upload_ood_image(current_step=iteration, energy_map=focus_area[:upload_img_num],
                                      img_number=upload_img_num, data_name=data_name, measure_way=measure_way)
        del curr_info
    if engine.distributed:
        dist.barrier()

    return


def valid_epoch(model, engine, test_set, my_wandb, evaluator=None, logger=None):
    if engine.local_rank <= 0:
        logger.info("validating {} dataset ...".format("cityscapes"))

    dir_name = "ours"
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    path_name = os.path.join(dir_name, "cityscapes_test")
    if not os.path.exists(path_name):
        os.mkdir(path_name)

    model.eval()
    curr_rank = max(0, engine.local_rank)
    stride = int(numpy.ceil(len(test_set) / engine.gpus))
    e_record = min((curr_rank + 1) * stride, len(test_set))
    shred_list = list(range(curr_rank * stride, e_record))

    curr_info = {}

    tbar = tqdm(shred_list, ncols=137, leave=True) if curr_rank <= 0 else shred_list

    total_inter, total_union = 0, 0
    total_correct, total_label = 0, 0

    with torch.no_grad():
        for idx in tbar:
            img, label = test_set[idx]
            img = img.permute(1, 2, 0).numpy()
            pred = evaluator(img, model)
            correct, labeled, inter, union = eval_metrics(pred.unsqueeze(0), label, num_classes=19, ignore_index=255)
            total_inter, total_union = total_inter + inter, total_union + union
            total_correct, total_label = total_correct + correct, total_label + labeled

        # deeplabv3+ current mIoU is 0.90687, mAcc is 0.98163
        # rpl.code current mIoU is 0.90687, mAcc is 0.98163
        # rpl_corocl.code current mIoU is 0.90687, mAcc is 0.98163
        # dense-hybrid current mIoU is 0.9050, mAcc is 0.9816
        # meta-ood current mIoU is 0.9005741103115877, mAcc is 0.9802561907901769
        # PEBAL current mIoU is 0.8959019124839167, mAcc is 0.9791710976759205
        if engine.distributed:
            total_inter = torch.tensor(total_inter, device=engine.local_rank)
            total_union = torch.tensor(total_union, device=engine.local_rank)
            total_label = torch.tensor(total_label, device=engine.local_rank)
            total_correct = torch.tensor(total_correct, device=engine.local_rank)

            dist.all_reduce(total_inter, dist.ReduceOp.SUM)
            dist.all_reduce(total_union, dist.ReduceOp.SUM)
            dist.all_reduce(total_correct, dist.ReduceOp.SUM)
            dist.all_reduce(total_label, dist.ReduceOp.SUM)

            total_inter, total_union = total_inter.cpu().numpy(), total_union.cpu().numpy()
            total_correct, total_label = total_correct.cpu().numpy(), total_label.cpu().numpy()

        pix_acc = 1.0 * total_correct / (numpy.spacing(1) + total_label)
        iou = 1.0 * total_inter / (numpy.spacing(1) + total_union)
        m_iou = iou.mean().item()
        m_acc = pix_acc.item()

        curr_info['inlier_miou'] = m_iou
        curr_info['inlier_macc'] = m_acc

    if my_wandb is not None and engine.local_rank <= 0:
        logger.critical("current mIoU is {:.5f}, mAcc is {:.5f}".format(curr_info['inlier_miou'],
                                                                        curr_info['inlier_macc']))
        my_wandb.upload_wandb_info(info_dict=curr_info, current_step=0, measure_way="inlier")

    return


def final_test_inlier(model, engine, test_set, output_dir, evaluator=None, logger=None):

    palette = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 111, 74, 0, 81, 0, 81, 128,
               64, 128, 244, 35, 232, 250, 170, 160, 230, 150, 140, 70, 70, 70, 102, 102,
               156, 190, 153, 153, 180, 165, 180, 150, 100, 100, 150, 120, 90, 153, 153,
               153, 153, 153, 153, 250, 170, 30, 220, 220, 0, 107, 142, 35, 152, 251, 152,
               70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70, 0, 60, 100, 0, 0,
               90, 0, 0, 110, 0, 80, 100, 0, 0, 230, 119, 11, 32, 0, 0, 142]

    def colorize_mask(mask):
        """
        Colorize a segmentation mask.
        """
        # mask: numpy array of the mask
        new_mask = Image.fromarray(mask.astype(numpy.uint8)).convert('P')
        new_mask.putpalette(palette)
        return new_mask

    def decode_train_id(result):
        train_id_tabel = test_set.train_id2label
        labels_ = numpy.unique(result)
        canvas_ = numpy.zeros_like(result)
        for cls in labels_:
            canvas_[result == cls] = train_id_tabel[cls].id
        return canvas_

    if engine.local_rank <= 0:
        logger.info("inference testing inlier images: {} dataset ...".format("cityscapes"))
    model.eval()
    curr_rank = max(0, engine.local_rank)
    stride = int(numpy.ceil(len(test_set) / engine.gpus))
    e_record = min((curr_rank + 1) * stride, len(test_set))
    shred_list = list(range(curr_rank * stride, e_record))

    tbar = tqdm(shred_list, ncols=137, leave=True) if curr_rank <= 0 else shred_list

    with torch.no_grad():
        for idx in tbar:
            img, index = test_set[idx]
            img = img.permute(1, 2, 0).numpy()
            output = torch.argmax(evaluator(img, model), dim=0).cpu().numpy()
            output = decode_train_id(output)
            output = colorize_mask(output)
            output.save(os.path.join(output_dir, index))
    return


def all_gather_samples(x_, y_, padding_value=666, local_rank=-1):
    # 1). gather the maximum size of the grab feature
    local_list_length = torch.tensor([len(x_)]).to(local_rank)
    global_list_length = [torch.tensor([0]).to(local_rank) for _ in range(dist.get_world_size())]

    dist.all_gather(global_list_length, local_list_length.clone(), async_op=False)
    global_list_length = max(global_list_length)

    x_ = torch.tensor(x_).squeeze().to(local_rank)
    y_ = torch.tensor(y_).squeeze().to(local_rank)
    # 2). padding with specific value
    if local_list_length < global_list_length:
        x_ = torch.cat([x_, torch.full((global_list_length - local_list_length, x_.shape[1], x_.shape[2]),
                                       fill_value=padding_value).to(local_rank)], dim=0)

        y_ = torch.cat([y_, torch.full((global_list_length - local_list_length, y_.shape[1], y_.shape[2]),
                                       fill_value=padding_value).to(local_rank)], dim=0)

    # 3). grab global lists with maximum size
    global_x_ = [torch.zeros_like(x_, device=x_.device) for _ in range(dist.get_world_size())]
    global_y_ = [torch.zeros_like(y_, device=x_.device) for _ in range(dist.get_world_size())]

    dist.all_gather(global_x_, x_.clone(), async_op=False)
    dist.all_gather(global_y_, y_.clone(), async_op=False)
    global_x_ = torch.cat(global_x_)
    global_y_ = torch.cat(global_y_)

    # 4). eliminate padding value
    msk = ~torch.eq(global_x_, padding_value).flatten(start_dim=1).all(dim=1)
    global_x_ = global_x_[msk, :]
    global_y_ = global_y_[msk, :]
    assert ~torch.eq(global_x_, padding_value).any() and ~torch.eq(global_y_, padding_value).any() \
           and ~torch.eq(global_x_, padding_value).any(), "gather padding value, exit."

    return global_x_.cpu().numpy(), global_y_.cpu().numpy()
