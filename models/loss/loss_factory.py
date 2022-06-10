import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def _make_input(t, requires_grad=False, device=torch.device('cpu')):
    """Make zero inputs for AE loss.

    Args:
        t (torch.Tensor): input
        requires_grad (bool): Option to use requires_grad.
        device: torch device

    Returns:
        torch.Tensor: zero input.
    """
    inp = torch.autograd.Variable(t, requires_grad=requires_grad)
    inp = inp.sum()
    inp = inp.to(device)

    return inp


class HeatmapsLoss(nn.Module):
    def __init__(self, beta=0.55, gamma=0.01):
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.criterion = nn.MSELoss(reduction='none')

    def forward(self, preds, targets, scales, masks=None):
        total_loss = 0
        for i, (pred, target) in enumerate(zip(preds, targets)):
            target = target ** (self.beta / scales[i]) ** 2
            loss = self.criterion(pred, target)

            pos_like = (1 - torch.log(target)) ** (-1 * self.gamma)
            weight = pos_like * torch.abs(1 - pred) + (1 - pos_like) * torch.abs(pred)
            loss = loss * weight

            if masks is not None:
                loss = loss * masks[i][:, None, :, :]

            total_loss += loss.mean(dim=3).mean(dim=2).mean(dim=1).mean(dim=0)

        return total_loss


class AssociativeEmbeddingLoss(nn.Module):
    def __init__(self, scale_res):
        super().__init__()
        self.scale_res = scale_res
        self.scale_dist = nn.Parameter(torch.arange(0.5 / scale_res, 1.0, 1 / scale_res), requires_grad=False)

    def embed_scales(self, box_scales):
        box_scales = box_scales.unsqueeze(1).expand(box_scales.shape[0], self.scale_res)
        scale_gap = torch.abs(box_scales - self.scale_dist[None, :])
        scale_embed = 1 / (scale_gap + 1e-10)
        scale_embed = F.normalize(scale_embed, p=2, dim=1)

        return scale_embed

    def singleTagLoss(self, pred_tag, joints, box_scales):
        """Associative embedding loss for one image.

        Note:
            - heatmaps weight: W
            - heatmaps height: H
            - max_num_people: M
            - num_keypoints: K

        Args:
            pred_tag (torch.Tensor[KxHxW,1]): tag of output for one image.
            joints (torch.Tensor[M,K,2]): joints information for one image.
        """
        tags = []
        pull = 0
        d_scale = 0
        valid_inds = np.where(joints[..., -1].sum(axis=-1))[0]
        scale_embeddings = self.embed_scales(box_scales)

        for ind in valid_inds:
            joints_per_person = joints[ind]
            tgt_vec = scale_embeddings[ind]
            tmp = []
            for joint in joints_per_person:
                if joint[1] > 0:
                    tmp.append(pred_tag[joint[0]])
            if len(tmp) == 0:
                continue
            tmp = torch.stack(tmp)
            tags.append(torch.mean(tmp, dim=0))
            pull = pull + torch.mean((tmp - tags[-1].expand_as(tmp))**2)

            pred_vec = F.normalize(tags[-1].abs(), p=2, dim=0)
            cosine = (tgt_vec * pred_vec).sum()
            d_scale = d_scale + (1 - cosine)

        num_tags = len(tags)
        if num_tags == 0:
            return (
                _make_input(torch.zeros(1).float(), device=pred_tag.device),
                _make_input(torch.zeros(1).float(), device=pred_tag.device),
                _make_input(torch.zeros(1).float(), device=pred_tag.device))
        elif num_tags == 1:
            return (_make_input(
                torch.zeros(1).float(), device=pred_tag.device), pull, d_scale)

        tags = torch.stack(tags)

        size = (num_tags, num_tags, self.scale_res)
        A = tags.expand(*size)
        B = A.permute(1, 0, 2).contiguous()

        diff = torch.square(A - B).sum(dim=-1)
        not_diagonal = torch.eye(num_tags).to(diff.device) != 1
        diff = diff[not_diagonal]

        push = torch.exp(-diff).mean()
        push_loss = push * 0.5
        pull_loss = pull / num_tags
        scale_loss = d_scale / num_tags

        return push_loss, pull_loss, scale_loss

    def forward(self, tags, joints, box_scales):
        """Accumulate the tag loss for each image in the batch.

        Note:
            - batch_size: N
            - heatmaps weight: W
            - heatmaps height: H
            - max_num_people: M
            - num_keypoints: K

        Args:
            tags (torch.Tensor[N,K,H,W]): tag channels of output.
            joints (torch.Tensor[N,M,K,2]): joints information.
            box_scales (torch.Tensor[N,M]): bounding box scales.
        """
        pushes, pulls, d_scales = [], [], []
        joints = joints.cpu().data.numpy()
        batch_size = tags.size(0)
        tags = tags.reshape([batch_size, -1, self.scale_res])
        for i in range(batch_size):
            push, pull, d_scale = self.singleTagLoss(tags[i], joints[i], box_scales[i])
            pushes.append(push)
            pulls.append(pull)
            d_scales.append(d_scale)

        pull_loss = torch.stack(pulls, dim=0).mean()
        push_loss = torch.stack(pushes, dim=0).mean()
        scale_loss = torch.stack(d_scales, dim=0).mean()

        return pull_loss, push_loss, scale_loss
