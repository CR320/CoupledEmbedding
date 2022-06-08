import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=in_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out + identity)

        return out


class HigherAssociativeEmbeddingHead(nn.Module):
    """Heatmaps for joints detection.
    Args:
        in_channels (int): Number of input channels.
        num_keys (int): Number of joint keys.
    """

    def __init__(self,
                 in_channels,
                 num_keys,
                 num_basic_blocks=4):
        super().__init__()
        self.num_keys = num_keys
        self.hms_pred_layers = nn.ModuleList([nn.Conv2d(in_channels=in_channels,
                                                        out_channels=num_keys,
                                                        kernel_size=1, stride=1, padding=0) for _ in range(2)])
        self.tms_pred_layers = nn.Conv2d(in_channels=in_channels, out_channels=num_keys,
                                         kernel_size=1, stride=1, padding=0)
        self.deconv_layers = nn.Sequential(nn.ConvTranspose2d(in_channels=in_channels + num_keys*2,
                                                              out_channels=in_channels,
                                                              kernel_size=4, stride=2, padding=1, bias=False),
                                           nn.BatchNorm2d(num_features=in_channels),
                                           nn.ReLU(inplace=True))
        self.blocks = nn.Sequential(*[BasicBlock(in_channels=in_channels, out_channels=in_channels)
                                      for _ in range(num_basic_blocks)])
        self.init_weights()

    def forward(self, x):
        """Forward function."""
        if isinstance(x, list):
            x = x[0]

        heatmaps = list()
        hms_low = self.hms_pred_layers[0](x)
        heatmaps.append(hms_low)
        tagmaps = self.tms_pred_layers(x)

        cat_x = torch.cat([x, hms_low, tagmaps], dim=1)
        cat_x = self.deconv_layers(cat_x)

        cat_x = self.blocks(cat_x)
        hms_high = self.hms_pred_layers[1](cat_x)
        heatmaps.append(hms_high)

        return heatmaps, tagmaps

    def init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, val=1)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
