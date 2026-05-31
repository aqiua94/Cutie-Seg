import torch
import torch.nn.functional as F


def warp_feature(feature: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Warp a feature map with pixel-space motion vectors.

    Args:
        feature: Tensor shaped [B, C, H, W].
        flow: Tensor shaped [B, H_mv, W_mv, 2], where channel 0 is x displacement
            and channel 1 is y displacement in the source image pixel space.
    """
    if flow is None:
        return feature

    if flow.ndim != 4 or flow.shape[-1] != 2:
        raise ValueError(f'flow must have shape [B, H, W, 2], got {tuple(flow.shape)}')

    b, _, h, w = feature.shape
    flow = flow.to(device=feature.device, dtype=feature.dtype).permute(0, 3, 1, 2)

    scale_y = h / flow.shape[-2]
    scale_x = w / flow.shape[-1]
    flow = flow.clone()
    flow[:, 0] *= scale_x
    flow[:, 1] *= scale_y
    flow = F.interpolate(flow, size=(h, w), mode='bilinear', align_corners=False)

    yy, xx = torch.meshgrid(torch.arange(h, device=feature.device, dtype=feature.dtype),
                            torch.arange(w, device=feature.device, dtype=feature.dtype),
                            indexing='ij')
    grid = torch.stack((xx, yy), dim=0).unsqueeze(0).expand(b, -1, -1, -1) + flow
    grid_x = 2.0 * grid[:, 0] / max(w - 1, 1) - 1.0
    grid_y = 2.0 * grid[:, 1] / max(h - 1, 1) - 1.0
    grid = torch.stack((grid_x, grid_y), dim=-1)

    return F.grid_sample(feature, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
