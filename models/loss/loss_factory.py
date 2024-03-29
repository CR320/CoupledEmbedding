import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ScaleWiseHeatmapsLoss(nn.Module):
    def __init__(self, with_focal, beta, gamma=0.01):
        super().__init__()
        self.with_focal = with_focal
        self.beta = beta
        self.gamma = gamma
        self.criterion = nn.MSELoss(reduction='none')

    def forward(self, preds, targets, scales, masks=None):
        total_loss = 0
        for i, (pred, target) in enumerate(zip(preds, targets)):
            target = target ** (self.beta / scales[i]) ** 2
            loss = self.criterion(pred, target)

            if self.with_focal:
                pos_like = (1 - torch.log(target)) ** (-1 * self.gamma)
                weight = pos_like * torch.abs(1 - pred) + (1 - pos_like) * torch.abs(pred)
                loss = loss * weight

            if masks is not None:
                loss = loss * masks[i][:, None, :, :]

            total_loss += loss.mean(dim=3).mean(dim=2).mean(dim=1).mean(dim=0)

        return total_loss


class RegularizedAssociativeEmbeddingLoss(nn.Module):
    def __init__(self, scale_res):
        super().__init__()
        self.scale_res = scale_res
        self.scale_dist = nn.Parameter(torch.arange(0.5 / scale_res, 1.0, 1 / scale_res), requires_grad=False)

    @ staticmethod
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

    def embed_scales(self, box_scales):
        box_scales = box_scales.unsqueeze(1).expand(box_scales.shape[0], self.scale_res)
        scale_gap = torch.abs(box_scales - self.scale_dist[None, :])
        scale_embeds = 1 / (scale_gap + 1e-12)
        scale_embeds = F.normalize(scale_embeds, p=2, dim=1)

        return scale_embeds

    def singleTagLoss(self, pred_tag, joints, box_scales):
        tags = []
        pull = 0
        d_scale = 0
        valid_inds = np.where(joints[..., -1].sum(axis=-1))[0]
        scale_embeds = self.embed_scales(box_scales)

        for ind in valid_inds:
            joints_per_person = joints[ind]
            tmp = []
            for joint in joints_per_person:
                if joint[1] > 0:
                    tmp.append(pred_tag[joint[0]])
            if len(tmp) == 0:
                continue
            tmp = torch.stack(tmp)
            tags.append(torch.mean(tmp, dim=0))
            pull = pull + torch.mean(torch.square(tmp - tags[-1].expand_as(tmp)).sum(-1))

            pred_embed = F.normalize(tags[-1].abs(), p=2, dim=0)
            tgt_embed = scale_embeds[ind]
            cosine = (pred_embed * tgt_embed).sum()
            d_scale = d_scale + (1 - cosine)

        num_tags = len(tags)
        if num_tags == 0:
            return (
                self._make_input(torch.zeros(1).float(), device=pred_tag.device),
                self._make_input(torch.zeros(1).float(), device=pred_tag.device),
                self._make_input(torch.zeros(1).float(), device=pred_tag.device))
        elif num_tags == 1:
            return (self._make_input(
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
