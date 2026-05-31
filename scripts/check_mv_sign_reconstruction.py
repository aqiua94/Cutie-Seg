import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from cutie.dataset.compressed_davis_dataset import CompressedDAVISDataset


def save_rgb(tensor: torch.Tensor, path: Path) -> None:
    array = (tensor.permute(1, 2, 0).detach().cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
    Image.fromarray(array).save(path)


def warp(src: torch.Tensor, grid_pix: torch.Tensor) -> torch.Tensor:
    _, height, width = src.shape
    grid = grid_pix.clone()
    grid[..., 0] = grid[..., 0] / (width - 1) * 2 - 1
    grid[..., 1] = grid[..., 1] / (height - 1) * 2 - 1
    return F.grid_sample(
        src.unsqueeze(0),
        grid.unsqueeze(0),
        mode='bilinear',
        padding_mode='border',
        align_corners=True,
    ).squeeze(0)


def find_clip_index(dataset: CompressedDAVISDataset, sequence: str, start_frame: str | None) -> int:
    fallback = None
    for idx, clip in enumerate(dataset.clips):
        if clip[0]['sequence'] != sequence:
            continue
        if fallback is None:
            fallback = idx
        if start_frame is not None and clip[0]['frame'] == start_frame:
            return idx
    if fallback is not None:
        return fallback
    raise RuntimeError(f'No clip found for sequence={sequence!r}')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--davis-root', type=Path, default=Path('data/DAVIS/2017/trainval'))
    parser.add_argument('--compressed-root', type=Path, default=Path('data/DAVIS/2017/trainval/compressed/3M-GOP12'))
    parser.add_argument('--output-dir', type=Path, default=Path('output/sanity/mv_sign_check'))
    parser.add_argument('--sequence', default='drift-turn')
    parser.add_argument('--start-frame', default='00048.jpg')
    parser.add_argument('--t', type=int, default=4)
    parser.add_argument('--size', type=int, default=480)
    parser.add_argument('--resize-mode', choices=['crop', 'square'], default='crop')
    parser.add_argument('--seq-length', type=int, default=8)
    parser.add_argument('--seed', type=int, default=14159265)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset = CompressedDAVISDataset(
        args.davis_root,
        args.compressed_root,
        seq_length=args.seq_length,
        size=args.size,
        max_num_obj=3,
        seed=args.seed,
        resize_mode=args.resize_mode,
    )
    idx = find_clip_index(dataset, args.sequence, args.start_frame)
    sample = dataset[idx]
    if args.t <= 0 or args.t >= sample['rgb'].shape[0]:
        raise ValueError(f'--t must be in [1, {sample["rgb"].shape[0] - 1}]')

    i_frame = sample['rgb'][0].float()
    p_frame = sample['rgb'][args.t].float()
    mv = sample['mv'][args.t].float()
    _, height, width = i_frame.shape

    yy, xx = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
    base = torch.stack([xx, yy], dim=-1).float()

    warped_plus = warp(i_frame, base + mv)
    warped_minus = warp(i_frame, base - mv)

    mse_plus = ((warped_plus - p_frame) ** 2).mean().item()
    mse_minus = ((warped_minus - p_frame) ** 2).mean().item()
    correct = '+mv' if mse_plus < mse_minus else '-mv'

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_rgb(i_frame, args.output_dir / 'I.png')
    save_rgb(p_frame, args.output_dir / 'P_target.png')
    save_rgb(warped_plus, args.output_dir / 'warp_plus.png')
    save_rgb(warped_minus, args.output_dir / 'warp_minus.png')

    frames = sample['info']['frames']
    mag = torch.linalg.vector_norm(mv, dim=-1)
    lines = [
        f'sequence={sample["info"]["name"]}',
        f'frames={frames}',
        f't={args.t} target_frame={frames[args.t]}',
        f'size={args.size} resize_mode={args.resize_mode} seq_length={args.seq_length}',
        f'mv_mean={mag.mean().item():.4f} mv_p95={mag.quantile(0.95).item():.4f} mv_max={mag.max().item():.4f}',
        f'MSE(warped_+mv, P_t) = {mse_plus:.6f}',
        f'MSE(warped_-mv, P_t) = {mse_minus:.6f}',
        f'correct_sign={correct}',
    ]
    report = '\n'.join(lines) + '\n'
    (args.output_dir / 'report.txt').write_text(report)
    print(report)


if __name__ == '__main__':
    main()
