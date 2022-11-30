import argparse
from collections import OrderedDict
import torch.optim
from config.config import config
from dataset.training.cityscapes import Cityscapes
from dataset.validation.fishyscapes import Fishyscapes
from dataset.validation.lost_and_found import LostAndFound
from dataset.validation.road_anomaly import RoadAnomaly
from dataset.validation.segment_me_if_you_can import SegmentMeIfYouCan
from engine.engine import Engine
from model.network import Network
from utils.img_utils import Compose, Normalize, ToTensor
from utils.wandb_upload import *
from utils.logger import *
from engine.evaluator import SlidingEval
from valid import valid_anomaly, valid_epoch, final_test_inlier


def get_anomaly_detector(ckpt_path):
    """
    Get Network Architecture based on arguments provided
    """
    ckpt_name = ckpt_path
    model = Network(config.num_classes)
    state_dict = torch.load(ckpt_name)
    state_dict = state_dict['model'] if 'model' in state_dict.keys() else state_dict
    state_dict = state_dict['model_state'] if 'model_state' in state_dict.keys() else state_dict
    state_dict = state_dict['state_dict'] if 'state_dict' in state_dict.keys() else state_dict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        new_state_dict[name] = v
    state_dict = new_state_dict
    model.load_state_dict(state_dict, strict=True)
    return model


def main(gpu, ngpus_per_node, config, args):
    args.local_rank = gpu
    logger = logging.getLogger("ours")
    logger.propagate = False

    engine = Engine(custom_arg=args, logger=logger,
                    continue_state_object=config.pretrained_weight_path)

    transform = Compose([ToTensor(), Normalize(config.image_mean, config.image_std)])

    cityscapes_val = Cityscapes(root=config.city_root_path, split="val", transform=transform)
    cityscapes_test = Cityscapes(root=config.city_root_path, split="test", transform=transform)
    evaluator = SlidingEval(config, device=0 if engine.local_rank < 0 else engine.local_rank)
    fishyscapes_ls = Fishyscapes(split='LostAndFound', root=config.fishy_root_path, transform=transform)
    fishyscapes_static = Fishyscapes(split='Static', root=config.fishy_root_path, transform=transform)
    segment_me_anomaly = SegmentMeIfYouCan(split='road_anomaly', root=config.segment_me_root_path, transform=transform)
    segment_me_obstacle = SegmentMeIfYouCan(split='road_obstacle', root=config.segment_me_root_path,
                                            transform=transform)
    road_anomaly = RoadAnomaly(root=config.road_anomaly_root_path, transform=transform)
    # lost_and_found = LostAndFound(root=config.lost_and_found_root_path, transform=transform)
    model = get_anomaly_detector(config.rpl_corocl_weight_path)
    vis_tool = Tensorboard(config=config)

    if engine.distributed:
        torch.cuda.set_device(engine.local_rank)
        model.cuda(engine.local_rank)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[engine.local_rank],
                                                          find_unused_parameters=True)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

    model.eval()
    """
    # 1). we currently only support single gpu valid for the cityscapes sliding validation, and it 
    # might take long time, feel free to uncomment it. (we'll have to use the sliding eval. to achieve 
      the performance reported in the GitHub. )
    # 2). we follow Meta-OoD to use single scale validation for OoD dataset, for fair comparison.
    """
    valid_anomaly(model=model, engine=engine, iteration=0, test_set=segment_me_anomaly,
                  data_name='segment_me_anomaly', my_wandb=vis_tool, logger=logger,
                  measure_way=config.measure_way)

    valid_anomaly(model=model, engine=engine, iteration=0, test_set=segment_me_obstacle,
                  data_name='segment_me_obstacle', my_wandb=vis_tool, logger=logger,
                  measure_way=config.measure_way)

    valid_anomaly(model=model, engine=engine, iteration=0, test_set=fishyscapes_static,
                  data_name='Fishyscapes_static', my_wandb=vis_tool, logger=logger,
                  measure_way=config.measure_way)

    valid_anomaly(model=model, engine=engine, iteration=0, test_set=fishyscapes_ls,
                  data_name='Fishyscapes_ls', my_wandb=vis_tool, logger=logger,
                  measure_way=config.measure_way)

    valid_anomaly(model=model, engine=engine, iteration=0, test_set=road_anomaly, data_name='road_anomaly',
                  my_wandb=vis_tool, logger=logger, measure_way=config.measure_way)

    valid_epoch(model, engine, cityscapes_val, vis_tool, evaluator=evaluator, logger=logger)

    # please note: final_test_inlier function only produce the prediction and encode it with 33 categories;
    # you'll need to submit to https://www.cityscapes-dataset.com and receive the test results.
    # final_test_inlier(model=model.module if engine.distributed else model,
    #                   engine=engine, test_set=cityscapes_test, output_dir="./",
    #                   evaluator=evaluator, logger=logger)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Anomaly Segmentation')
    parser.add_argument('--gpus', default=1,
                        type=int,
                        help="gpus in use")
    parser.add_argument("--ddp", action="store_true",
                        help="distributed data parallel training or not;"
                             "MUST SPECIFIED")
    parser.add_argument('-l', '--local_rank', default=-1,
                        type=int,
                        help="distributed or not")
    parser.add_argument('-n', '--nodes', default=1,
                        type=int,
                        help="distributed or not")

    args = parser.parse_args()
    args.world_size = args.nodes * args.gpus

    # we enforce the flag of ddp if gpus >= 2;
    args.ddp = True if args.world_size > 1 else False
    if args.gpus <= 1:
        main(-1, 1, config=config, args=args)
    else:
        torch.multiprocessing.spawn(main, nprocs=args.gpus, args=(args.gpus, config, args))
