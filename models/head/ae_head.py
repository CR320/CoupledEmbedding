import torch.nn as nn


class AssociativeEmbeddingHead(nn.Module):
    """Heatmaps for joints detection & grouping.
    Args:
        in_channels (int): Number of input channels.
        num_keys (int): Number of joint types.
    """

    def __init__(self,
                 in_channels,
                 num_keys,
                 scale_res):
        super().__init__()
        self.num_keys = num_keys
        self.scale_res = scale_res
        self.hms_pred_layers = nn.Conv2d(in_channels=in_channels, out_channels=num_keys,
                                         kernel_size=1, stride=1, padding=0)
        self.tms_pred_layers = nn.Conv2d(in_channels=in_channels, out_channels=num_keys * scale_res,
                                         kernel_size=1, stride=1, padding=0)
        self.init_weights()

    def forward(self, x):
        """Forward function."""
        if isinstance(x, list):
            x = x[0]

        heatmaps = [self.hms_pred_layers(x)]
        tagmaps = self.tms_pred_layers(x)

        # reshape tagmaps
        num_b, _, num_h, num_w = tagmaps.shape[0:4]
        tagmaps = tagmaps.view(num_b, self.num_keys, self.scale_res, num_h, num_w)
        tagmaps = tagmaps.permute(0, 1, 3, 4, 2).contiguous()

        return heatmaps, tagmaps

    def init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
