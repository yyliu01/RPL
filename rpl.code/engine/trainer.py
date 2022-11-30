import os

import torch
from tqdm import tqdm


class Trainer:
    """
    loss_1 -> gambler loss; loss_2 -> energy loss
    lr_scheduler -> cosine;
    """

    def __init__(self, engine, loss1, tensorboard, validator, lr_scheduler, ckpt_dir, energy_weight):
        self.engine = engine
        self.energy_loss = loss1
        self.energy_weight = energy_weight

        self.lr_scheduler = lr_scheduler
        self.validator = validator
        self.saved_dir = ckpt_dir
        self.tensorboard = tensorboard

    def train(self, model, epoch, train_sampler, train_loader, optimizer):
        model.train()
        self.freeze_model_parameters(model)

        if self.engine.distributed:
            train_sampler.set_epoch(epoch=epoch)

        loader_len = len(train_loader)
        tbar = range(loader_len)
        train_loader = iter(train_loader)

        for batch_idx in tbar:
            city_imgs, city_targets, city_mix_imgs, city_mix_targets, ood_imgs, ood_targets = next(train_loader)

            city_mix_imgs = city_mix_imgs.cuda(non_blocking=True)
            city_mix_targets = city_mix_targets.cuda(non_blocking=True)

            optimizer.zero_grad()
            curr_idx = epoch * loader_len + batch_idx
            self.engine.update_iteration(epoch, curr_idx)

            city_vanilla_logits, city_mix_logits = model(city_mix_imgs)
            # energy training
            loss_dict = self.energy_loss(logits=city_mix_logits, targets=city_mix_targets.clone(),
                                         vanilla_logits=city_vanilla_logits)
            inlier_loss = loss_dict["entropy_part"] + loss_dict["reg"]
            outlier_loss = loss_dict["energy_part"] * self.energy_weight

            loss = inlier_loss + outlier_loss
            loss.backward()
            optimizer.step()

            # free cache memos for fixed params.
            torch.cuda.empty_cache()

            # update learning rate
            current_lr = self.lr_scheduler.get_lr(cur_iter=curr_idx)
            for i, opt_group in enumerate(optimizer.param_groups[:2]):
                opt_group['lr'] = current_lr * 10.

            for i, opt_group in enumerate(optimizer.param_groups[2:]):
                opt_group['lr'] = current_lr

            curr_info = {}
            if self.engine.local_rank <= 0:
                curr_info['lr'] = current_lr
                curr_info['entropy_loss (inlier pixels)'] = inlier_loss.cpu().item()
                curr_info['energy_loss (outlier pixels)'] = outlier_loss.cpu().item()
                curr_info['contrastive_loss'] = 0
                self.tensorboard.upload_wandb_info(current_step=curr_idx, info_dict=curr_info, measure_way="entropy")

                if curr_idx % 10 == 0:
                    self.tensorboard.upload_wandb_image(images=city_mix_imgs, ground_truth=city_mix_targets,
                                                        ood_images=None, ood_gt=None,
                                                        ood_prediction=None,
                                                        city_prediction=city_mix_logits)

                print("epoch={}, iter=[{}/{}] | "
                      "entropy: {:.3f} energy: {:.3f}, contrastive: {:.3f}".format(epoch, batch_idx, loader_len,
                                                                                   inlier_loss.cpu().item(),
                                                                                   outlier_loss.cpu().item(), .0))

            if curr_idx % self.validator.eval_iter == 0 and epoch >= 5:
                self.validator.run(model, self.engine, curr_idx, self.tensorboard)
                self.freeze_model_parameters(model)
                if self.engine.local_rank <= 0:
                    ckpt_name = 'iter_{}'.format(str(curr_idx))
                    print('saving a checkpoint in iter{} to {} ...'.format(str(curr_idx),
                                                                           os.path.join(self.saved_dir, ckpt_name)))
                    self.engine.save_and_link_checkpoint(snapshot_dir=self.saved_dir, name=ckpt_name)

            del ood_imgs, city_mix_imgs, city_imgs
            del city_mix_logits, city_vanilla_logits
            del ood_targets, city_mix_targets, city_targets

    def freeze_model_parameters(self, curr_model):
        model = curr_model.module if self.engine.distributed else curr_model
        for name, param in model.named_parameters():
            if any([i in name for i in model.branch1.train_module_list]):
                param.requires_grad = True
            else:
                param.requires_grad = False

        model.branch1.eval()
        model.branch1.residual_block.train()

    def fetch_global_ood_num(self, local_ood_num):
        if self.engine.distributed:
            global_ood_num = [torch.tensor([0], device=self.engine.local_rank)
                              for _ in range(torch.distributed.get_world_size())]

            torch.distributed.all_gather(global_ood_num, local_ood_num.clone().detach(),
                                         async_op=False)
        else:
            global_ood_num = [local_ood_num]

        return global_ood_num
