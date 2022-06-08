import torch.nn as nn
from models.loss.loss_factory import HeatmapsLoss, AssociativeEmbeddingLoss


class Trainer(nn.Module):
    def __init__(self,
                 model,
                 loss_weights):
        super().__init__()
        self.net = model
        self.loss_weights = loss_weights
        self.hms_criterion = HeatmapsLoss()
        self.ae_criterion = AssociativeEmbeddingLoss()

    def forward(self, data):
        losses = dict()
        image = data['image']
        masks = data['masks']
        target_hms = data['target_hms']
        target_joints = data['joints'][0]

        pred_hms, pred_tms = self.net(image)
        losses['hms_loss'] = self.hms_criterion(pred_hms, target_hms, masks)
        losses['pull_loss'], losses['push_loss'] = self.ae_criterion(pred_tms, target_joints)

        # wighting loss value
        for k in self.loss_weights:
            assert k in losses
            losses[k] = losses[k] * self.loss_weights[k]

        return losses
