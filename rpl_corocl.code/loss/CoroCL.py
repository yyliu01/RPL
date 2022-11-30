import torch
from abc import ABC
import torch.nn as nn


class ContrastLoss(nn.Module, ABC):
    def __init__(self, engine, config=None):
        super(ContrastLoss, self).__init__()
        self.engine = engine
        self.temperature = 0.10
        self.ignore_idx = 255
        self.ood_idx = 254
        self.max_views = 512

    def forward(self, city_proj, city_gt, city_pred, ood_proj, ood_gt, ood_pred):
        city_gt = torch.nn.functional.interpolate(city_gt.unsqueeze(1).float(), size=city_proj.shape[2:],
                                                  mode='nearest').squeeze().long()

        ood_gt = torch.nn.functional.interpolate(ood_gt.unsqueeze(1).float(), size=ood_proj.shape[2:],
                                                 mode='nearest').squeeze().long()

        # normalise the embed results
        city_proj = torch.nn.functional.normalize(city_proj, p=2, dim=1)
        ood_proj = torch.nn.functional.normalize(ood_proj, p=2, dim=1)

        # randomly extract embed samples within a batch
        anchor_embeds, anchor_labels, contrs_embeds, contrs_labels = self.extraction_samples(city_proj, city_gt,
                                                                                             ood_proj, ood_gt)

        # calculate the CoroCL
        loss = self.info_nce(anchors_=anchor_embeds, a_labels_=anchor_labels.unsqueeze(1), contras_=contrs_embeds,
                             c_labels_=contrs_labels.unsqueeze(1)) if anchor_embeds.nelement() > 0 else \
            torch.tensor([.0], device=city_proj.device)

        return loss

    # The implementation of cross-image contrastive learning is based on:
    # https://github.com/tfzhou/ContrastiveSeg/blob/287e5d3069ce6d7a1517ddf98e004c00f23f8f99/lib/loss/loss_contrast.py
    def info_nce(self, anchors_, a_labels_, contras_, c_labels_):
        # calculates the binary mask: same category => 1, different categories => 0
        mask = torch.eq(a_labels_, torch.transpose(c_labels_, 0, 1)).float()

        # calculates the dot product
        anchor_dot_contrast = torch.div(torch.matmul(anchors_, torch.transpose(contras_, 0, 1)),
                                        self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # calculates the negative mask
        neg_mask = 1 - mask
        
        # avoid the self duplicate issue
        mask = mask.fill_diagonal_(0.)

        # sum the negative odot results
        neg_logits = torch.exp(logits) * neg_mask
        neg_logits = neg_logits.sum(1, keepdim=True)

        exp_logits = torch.exp(logits)

        # log_prob -> log(exp(x))-log(exp(x) + exp(y))
        # log_prob -> log{exp(x)/[exp(x)+exp(y)]}
        log_prob = logits - torch.log(exp_logits + neg_logits)

        # calculate the info-nce based on the positive samples (under same categories)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        return - mean_log_prob_pos.mean()

    def extraction_samples(self, city_embd, city_label, ood_embd, ood_label):
        # reformat the matrix
        city_embd = city_embd.flatten(start_dim=2).permute(0, 2, 1)
        city_label = city_label.flatten(start_dim=1)
        ood_embd = ood_embd.flatten(start_dim=2).permute(0, 2, 1)
        ood_label = ood_label.flatten(start_dim=1)

        # define different types of embeds
        city_positive = city_embd[city_label == self.ood_idx]
        city_negative = city_embd[(city_label != self.ood_idx) & (city_label != self.ignore_idx)]
        ood_positive = ood_embd[ood_label == self.ood_idx]
        ood_negative = ood_embd[(ood_label != self.ood_idx) & (ood_label != self.ignore_idx)]

        # define the number of choice
        sample_num = int(min(self.max_views, city_positive.shape[0], ood_positive.shape[0],
                             city_negative.shape[0], ood_negative.shape[0]))

        # randomly extract the anchor set with {city_ood, city_inlier}
        city_positive_anchor = city_positive[torch.randperm(city_positive.shape[0])][:sample_num]
        city_negative_anchor = city_negative[torch.randperm(city_negative.shape[0])][:sample_num]

        anchor_embed = torch.cat([city_positive_anchor, city_negative_anchor], dim=0)

        anchor_label = torch.cat([torch.empty(city_positive_anchor.shape[0],
                                              device=city_positive_anchor.device).fill_(1.),
                                  torch.empty(city_negative_anchor.shape[0],
                                              device=city_negative_anchor.device).fill_(0.)])

        # randomly extract the contras set with {city_ood, city_inlier, coco_ood, coco_inlier}
        city_positive_contras = city_positive_anchor.clone()
        city_negative_contras = city_negative_anchor.clone()
        ood_positive_contras = ood_positive[torch.randperm(ood_positive.shape[0])][:sample_num]
        ood_negative_contras = ood_negative[torch.randperm(ood_negative.shape[0])][:sample_num]

        contrs_embed = torch.cat([city_positive_contras, city_negative_contras,
                                  ood_positive_contras, ood_negative_contras], dim=0)

        contrs_label = torch.cat([torch.empty(city_positive_contras.shape[0],
                                              device=city_positive_contras.device).fill_(1.),
                                  torch.empty(city_negative_contras.shape[0],
                                              device=city_negative_contras.device).fill_(0.),
                                  torch.empty(ood_positive_contras.shape[0],
                                              device=ood_positive_contras.device).fill_(1.),
                                  torch.empty(ood_negative_contras.shape[0],
                                              device=ood_negative_contras.device).fill_(0.)])

        return anchor_embed, anchor_label, contrs_embed, contrs_label

