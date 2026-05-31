import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _local_attention_pytorch(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, kH: int,
                             kW: int) -> torch.Tensor:
    b, c, h, w = query.shape
    key_patches = F.unfold(key, kernel_size=(kH, kW), padding=(kH // 2, kW // 2))
    value_patches = F.unfold(value, kernel_size=(kH, kW), padding=(kH // 2, kW // 2))
    key_patches = key_patches.view(b, c, kH * kW, h, w)
    value_patches = value_patches.view(b, c, kH * kW, h, w)

    weights = (query.unsqueeze(2) * key_patches).sum(dim=1) / math.sqrt(c)
    weights = torch.softmax(weights, dim=1)
    return (value_patches * weights.unsqueeze(1)).sum(dim=2)


class CReFFBlock(nn.Module):
    """Cross Resolution Feature Fusion block adapted from AR-Seg.

    This implementation keeps the AR-Seg design: depthwise Q/K/V projections,
    local attention over a kH x kW window, and a residual connection from the
    upsampled LR feature.
    """

    def __init__(self, feat_dim: int = 256, kH: int = 7, kW: int = 7):
        super().__init__()
        self.lr_query_conv = nn.Conv2d(feat_dim,
                                       feat_dim,
                                       kernel_size=3,
                                       padding=1,
                                       groups=feat_dim)
        self.hr_key_conv = nn.Conv2d(feat_dim,
                                     feat_dim,
                                     kernel_size=3,
                                     padding=1,
                                     groups=feat_dim)
        self.hr_value_conv = nn.Conv2d(feat_dim,
                                       feat_dim,
                                       kernel_size=3,
                                       padding=1,
                                       groups=feat_dim)
        self.kH = kH
        self.kW = kW
        self.init_weight()

    def init_weight(self) -> None:
        for layer in self.children():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, a=1)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    def forward(self, hr_feat: torch.Tensor, lr_feat: torch.Tensor) -> torch.Tensor:
        _, _, h, w = hr_feat.shape
        lr_feat = F.interpolate(lr_feat, size=(h, w), mode='bilinear', align_corners=True)

        query = self.lr_query_conv(lr_feat)
        key = self.hr_key_conv(hr_feat)
        value = self.hr_value_conv(hr_feat)
        attention_result = _local_attention_pytorch(query, key, value, self.kH, self.kW)

        return lr_feat + attention_result
