import torch.nn as nn
import torch.nn.functional as F


class PoseDet(nn.Module):
    def __init__(self,
                 backbone,
                 head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, input, phase='train'):
        # backbone inference
        fea = self.backbone(input)

        # calculate heatmaps & tagmaps
        if phase == 'train':
            return self.head(fea)
        elif phase == 'inference':
            heatmaps, tagmaps = self.head(fea)
            if len(heatmaps) > 1:
                hms_low = heatmaps[0]
                hms_high = heatmaps[1]
                hms_low_resized = F.interpolate(hms_low,
                                                size=hms_high.shape[-2:],
                                                mode='bilinear', align_corners=False)
                heatmaps = (hms_low_resized + hms_high) / 2.
            else:
                heatmaps = heatmaps[0]
            return heatmaps, tagmaps
        else:
            raise ValueError('Wrong phase name')