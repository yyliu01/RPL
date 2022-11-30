import torch
import torch.nn.functional as func


# vanilla_logits := logits w/o att_ block
# logits := logits w/ att_ block
def disimilarity_entropy(logits, vanilla_logits, t=1.):
    n_prob = torch.clamp(torch.softmax(vanilla_logits, dim=1), min=1e-7)
    a_prob = torch.clamp(torch.softmax(logits, dim=1), min=1e-7)

    n_entropy = -torch.sum(n_prob * torch.log(n_prob), dim=1) / t
    a_entropy = -torch.sum(a_prob * torch.log(a_prob), dim=1) / t

    entropy_disimilarity = torch.nn.functional.mse_loss(input=a_entropy, target=n_entropy, reduction="none")
    assert ~torch.isnan(entropy_disimilarity).any(), print(torch.min(n_entropy), torch.max(a_entropy))

    return entropy_disimilarity


def energy_loss(logits, targets, vanilla_logits, out_idx=254, t=1.):
    out_msk = (targets == out_idx)
    void_msk = (targets == 255)

    pseudo_targets = torch.argmax(vanilla_logits, dim=1)
    outlier_msk = (out_msk | void_msk)
    entropy_part = func.cross_entropy(input=logits, target=pseudo_targets, reduction='none')[~outlier_msk]
    reg = disimilarity_entropy(logits=logits, vanilla_logits=vanilla_logits)[~outlier_msk]
    if torch.sum(out_msk) > 0:
        logits = logits.flatten(start_dim=2).permute(0, 2, 1)
        energy_part = torch.nn.functional.relu(torch.log(torch.sum(torch.exp(logits),
                                                                   dim=2))[out_msk.flatten(start_dim=1)]).mean()
    else:
        energy_part = torch.tensor([.0], device=targets.device)
    return {"entropy_part": entropy_part.mean(), "reg": reg.mean(), "energy_part": energy_part}


