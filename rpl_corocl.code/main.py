import argparse
import torch.optim
from config.config import config
from dataset.data_loader import get_mix_loader
# from dgx.download_to_pvc import *
from engine.engine import Engine
from engine.lr_policy import WarmUpPolyLR
from engine.trainer import Trainer
from loss.CoroCL import ContrastLoss as BatchContrast
from loss.PositiveEnergy import energy_loss
from model.network import Network
from utils.conv_2_5d import group_weight
from utils.img_utils import *
from utils.wandb_upload import *
from utils.logger import *
from valid import *


def seed_it(seed):
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.manual_seed(seed)


def declare_settings(config_file, logger, engine):
    logger.critical("distributed data parallel training: {}".format(str("on" if engine.distributed is True
                                                                        else "off")))

    logger.critical("gpus: {}, with batch_size [local]: {}".format(engine.world_size, config_file.batch_size))

    logger.critical("network architecture: {}, with ResNet {} backbone".format("deeplabv3+",
                                                                               config_file['pretrained_weight_path']
                                                                               .split('/')[-1].split('_')[0]))
    logger.critical("learning rate: other {}, training {} epochs".format(config_file['lr'],
                                                                         config_file['nepochs']))

    logger.critical("city-context image: {}x{}, ood-context image: {}x{}".format(config_file['city_image_height'],
                                                                                 config_file['city_image_width'],
                                                                                 config_file['ood_image_height'],
                                                                                 config_file['ood_image_width']))


def main(gpu, ngpus_per_node, config, args):
    args.local_rank = gpu
    logger = logging.getLogger("rpl+corocl")
    logger.propagate = False
    engine = Engine(custom_arg=args, logger=logger,
                    continue_state_object=config.pretrained_weight_path)

    if engine.local_rank <= 0:
        logger.critical(config)
        declare_settings(config_file=config, logger=logger, engine=engine)
        visual_tool = Tensorboard(config=config)
        """
        if args.gcloud:
            logger.critical("downloading fishyscapes dataset ...")
            download_ex5_dataset_unzip(data_dir="./fishyscapes", prefix="yy/exercise_5/dataset/",
                                       bucket_prefix="fishyscapes.zip", pvc=args.pvc)

            # logger.critical("downloading lost-and-found dataset ...")
            # download_ex5_dataset_unzip(data_dir="./lost_and_found", prefix="yy/exercise_5/dataset/",
            #                            bucket_prefix="lost_and_found.zip", pvc=args.pvc)

            logger.critical("downloading road-anomaly dataset ...")
            download_ex5_dataset_unzip(data_dir="./road_anomaly", prefix="yy/exercise_5/dataset/",
                                       bucket_prefix="road_anomaly.zip", pvc=args.pvc)

            logger.critical("downloading segment-me-if-u-can dataset ...")
            download_ex5_dataset_unzip(data_dir="./segment_me", prefix="yy/exercise_5/dataset/",
                                       bucket_prefix="segment_me.zip", pvc=args.pvc)

            logger.critical("downloading pretrained checkpoint ...")
            download_checkpoint("./ckpts/", 'pretrained_ckpts/'+config.pretrained_weight_path.split('/')[-1])

            logger.critical("downloading cityscapes dataset ...")
            download_ex5_dataset_unzip(data_dir="./city_scape", prefix="yy/exercise_5/dataset/",
                                       bucket_prefix="city.zip", pvc=args.pvc)

            logger.critical("downloading coco dataset ...")
            download_ex5_dataset_unzip(data_dir="./coco", prefix="yy/exercise_5/dataset/",
                                       bucket_prefix="coco.zip", pvc=args.pvc)
        """
    else:
        visual_tool = None

    seed_it(config.seed + engine.local_rank)
    model = Network(config.num_classes)
    contras_loss = BatchContrast(engine=engine, config=config)

    params_list = []
    # final mlp's lr is 10 times larger than others,
    # as it is trained from scratch.
    for module in model.branch1.residual_block.retrain_layers:
        params_list = group_weight(params_list, module, torch.nn.BatchNorm2d,
                                   config.lr * 10.)

    for module in model.branch1.residual_block.fine_tune_layers:
        params_list = group_weight(params_list, module, torch.nn.BatchNorm2d,
                                   config.lr)

    """
    selection of the opimizer
    """
    optimizer = torch.optim.Adam(params_list)

    # config lr policy
    base_lr = config.lr
    total_iteration = config.nepochs * config.niters_per_epoch
    lr_policy = WarmUpPolyLR(base_lr, config.lr_power, total_iteration,
                             config.niters_per_epoch * config.warm_up_epoch)
    validator = Validator(config=config, logger=logger)

    # define the trainer
    trainer = Trainer(engine=engine, loss1=energy_loss, loss2=contras_loss,
                      lr_scheduler=lr_policy, ckpt_dir=config.saved_dir, tensorboard=visual_tool, validator=validator,
                      energy_weight=config.energy_weight)

    if engine.distributed:
        torch.cuda.set_device(engine.local_rank)
        model.cuda(engine.local_rank)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[engine.local_rank],
                                                          find_unused_parameters=True)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

    # starting with the pre-trained weight from https://github.com/NVIDIA/semantic-segmentation/tree/sdcnet
    if engine.continue_state_object:
        engine.register_state(dataloader=None, model=model, optimizer=optimizer)
        engine.restore_checkpoint(extra_channel=False)

    if engine.local_rank <= 0:
        logger.info('training begin...')

    if engine.distributed:
        dist.barrier()

    for curr_epoch in range(engine.state.epoch, config.nepochs):
        train_loader, train_sampler = get_mix_loader(engine=engine, config=config, proba_factor=args.prob)

        engine.register_state(dataloader=train_loader, model=model, optimizer=optimizer)
        trainer.train(model=model, epoch=curr_epoch, train_sampler=train_sampler, train_loader=train_loader,
                      optimizer=optimizer)

        """ # for google cloud uploading
        if engine.local_rank <= 0 and args.gcloud:
            pvc_dir = os.path.join("yy", "exercise_5", "wider_resnet38", config.experiment_name)
            for ckpts in os.listdir(config.saved_dir):
                if "iter" not in ckpts:
                    continue
                upload_checkpoint(local_path=config.saved_dir, prefix=pvc_dir, checkpoint_filepath=ckpts)
            os.system("rm -rf {}/*.pth".format(config.saved_dir))
            print("remove temporal ckpt pth files.")
        """

    if engine.local_rank <= 0:
        visual_tool.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Anomaly Segmentation')
    parser.add_argument('--gpus', default=1, type=int, help="gpus in use")
    parser.add_argument('--local_rank', default=-1, type=int, help="distributed or not")
    parser.add_argument('--nodes', default=1, type=int, help="distributed or not")
    parser.add_argument('--pvc', action="store_true", help="pvc or not")
    parser.add_argument('--dgx', action="store_true", help="dgx or not")
    parser.add_argument('--gcloud', action="store_true", help="gcloud or not")
    parser.add_argument('--batch_size', default=8, type=int, help="batch in use")
    parser.add_argument('--prob', default=1., type=float, help="sample fetch-prob for OE")
    args = parser.parse_args()

    if args.pvc:
        config.city_root_path = "/pvc/city_scape"
        config.coco_root_path = "/pvc/coco"
        config.fishy_root_path = "/pvc/fishyscapes"
        config.segment_me_root_path = "/pvc/segment_me"
        config.road_anomaly_root_path = '/pvc/road_anomaly'
        config.segment_me_lf_root_path = '/pvc/segment_me_lf'
        config.lost_and_found_root_path = '/pvc/lost_and_found'

    args.world_size = args.nodes * args.gpus
    config.eval_iter = int(config.eval_iter / args.world_size)

    # we enforce the flag of ddp if gpus >= 2;
    args.ddp = True if args.world_size > 1 else False
    if args.gpus <= 1:
        main(-1, 1, config=config, args=args)
    else:
        torch.multiprocessing.spawn(main, nprocs=args.gpus, args=(args.gpus, config, args))
