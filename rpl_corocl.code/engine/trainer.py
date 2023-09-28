import os
import torchvision
import matplotlib.pyplot as plt
import numpy
import torch


class Trainer:
    """
    loss_1 -> energy loss; loss_2 -> contrastive loss
    lr_scheduler -> poly;
    """

    def __init__(self, engine, loss1, loss2, tensorboard, validator, lr_scheduler, ckpt_dir, energy_weight):
        self.engine = engine
        self.energy_loss = loss1
        self.contras_loss = loss2
        self.energy_weight = energy_weight

        self.saved_dir = ckpt_dir
        self.validator = validator
        self.tensorboard = tensorboard
        self.lr_scheduler = lr_scheduler

    def train(self, model, epoch, train_sampler, train_loader, optimizer):
        model.train()
        self.freeze_model_parameters(model)

        if self.engine.distributed:
            train_sampler.set_epoch(epoch=epoch)

        loader_len = len(train_loader)
        tbar = range(loader_len)
        train_loader = iter(train_loader)
        catch_anomaly = 0

        for batch_idx in tbar:
            curr_idx = epoch * loader_len + batch_idx
            city_imgs, city_targets, city_mix_imgs, city_mix_targets, ood_imgs, ood_targets = next(train_loader)
            city_mix_imgs, city_mix_targets = city_mix_imgs.cuda(non_blocking=True), city_mix_targets.cuda(non_blocking=True)
            ood_imgs, ood_targets = ood_imgs.cuda(non_blocking=True), ood_targets.cuda(non_blocking=True)
            optimizer.zero_grad()
            self.engine.update_iteration(epoch, curr_idx)
            ood_indices = [254 in i for i in city_mix_targets]
            global_ood_num = self.fetch_global_ood_num(local_ood_num=torch.tensor([sum(ood_indices)]).cuda())
            catch_anomaly += sum(global_ood_num).item()
            
            # the reason for such condition is provided in the issue below:
            # https://github.com/yyliu01/RPL/issues/2#issuecomment-1737459734
            if all([i >= 2 for i in global_ood_num]):
                input_data = torch.cat([city_mix_imgs, ood_imgs], dim=0)
                half_batch_size = int(input_data.shape[0] / 2)
                non_residual_logits, residual_logits, projects = model(input_data)

                city_vanilla_logits, city_mix_logits, city_proj = \
                    non_residual_logits[:half_batch_size], residual_logits[:half_batch_size], projects[:half_batch_size]

                ood_logits, ood_proj = residual_logits[half_batch_size:], projects[half_batch_size:]
                contras_loss = self.contras_loss(city_proj=city_proj[ood_indices],
                                                 city_gt=city_mix_targets[ood_indices],
                                                 city_pred=city_mix_logits[ood_indices],
                                                 ood_pred=ood_logits[ood_indices],
                                                 ood_proj=ood_proj[ood_indices], ood_gt=ood_targets[ood_indices])
            else:
                city_vanilla_logits, city_mix_logits, _ = model(city_mix_imgs)
                contras_loss = torch.tensor([.0], device=city_mix_logits.device)
                ood_logits = torch.tensor([.0], device=city_mix_logits.device)
                ood_proj, city_proj = torch.tensor([.0], device=city_mix_logits.device), torch.tensor([.0], device=city_mix_logits.device)
                
            # energy loss
            loss_dict = self.energy_loss(logits=city_mix_logits, targets=city_mix_targets.clone(),
                                         vanilla_logits=city_vanilla_logits)

            inlier_loss = loss_dict["entropy_part"] + loss_dict["reg"]
            outlier_loss = loss_dict["energy_part"] * self.energy_weight

            loss = inlier_loss + outlier_loss + contras_loss

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
                curr_info['contrastive_loss'] = contras_loss.cpu().item()
                self.tensorboard.upload_wandb_info(current_step=curr_idx, info_dict=curr_info)

                if curr_idx % 100 == 0:
                    self.tensorboard.upload_wandb_image(images=city_mix_imgs, ground_truth=city_mix_targets,
                                                        ood_images=None, ood_gt=None,
                                                        ood_prediction=ood_logits,
                                                        city_prediction=city_mix_logits)

                print("epoch={}, iter=[{}/{}] | entropy: {:.3f} energy: {:.3f}, contrastive: {:.3f}"
                      .format(epoch, batch_idx, loader_len, inlier_loss.cpu().item(),
                              outlier_loss.cpu().item(), contras_loss.cpu().item()))

            if curr_idx % self.validator.eval_iter == 0 and epoch >= 5:
                self.validator.run(model, self.engine, curr_idx, self.tensorboard)
                self.freeze_model_parameters(model)
                if self.engine.local_rank <= 0:
                    ckpt_name = 'iter_{}'.format(str(curr_idx))
                    print('saving a checkpoint in iter{} to {} ...'.format(str(curr_idx),
                                                                           os.path.join(self.saved_dir, ckpt_name)))
                    self.engine.save_and_link_checkpoint(snapshot_dir=self.saved_dir, name=ckpt_name)

            del ood_imgs, city_mix_imgs, city_imgs
            del ood_logits, city_mix_logits, city_vanilla_logits
            del ood_targets, city_mix_targets, city_targets
            del ood_proj, city_proj
            del loss_dict

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

