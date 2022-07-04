import torch.nn as nn
import torch.nn.functional as F
from models.loss.loss_factory import HeatmapsLoss, AssociativeEmbeddingLoss


class Trainer(nn.Module):
    def __init__(self,
                 model,
                 scale_res,
                 with_focal,
                 beta,
                 loss_weights):
        super().__init__()
        self.net = model
        self.loss_weights = loss_weights
        self.hms_criterion = HeatmapsLoss(with_focal, beta)
        self.ae_criterion = AssociativeEmbeddingLoss(scale_res)

    def forward(self, data):
        losses = dict()
        image = data['image']
        masks = data['masks']
        target_hms = data['target_hms']
        target_joints = data['joints'][0]
        target_scales = data['box_scales']

        pred_hms, pred_tms = self.net(image)

        scales = list()
        tags = pred_tms.detach()
        tags_norm = F.normalize(tags.abs(), p=1, dim=-1)
        scales.append((tags_norm * self.ae_criterion.scale_dist[None, None, None, None, :]).sum(dim=-1))
        if len(pred_hms) == 2:
            scales.append(F.interpolate(scales[0], size=pred_hms[1].shape[-2:], mode='bilinear', align_corners=False))

        losses['hms_loss'] = self.hms_criterion(pred_hms, target_hms, scales, masks)
        losses['pull_loss'], losses['push_loss'], losses['scale_loss'] = \
            self.ae_criterion(pred_tms, target_joints, target_scales)

        # wighting loss value
        for k in self.loss_weights:
            assert k in losses
            losses[k] = losses[k] * self.loss_weights[k]

        return losses
