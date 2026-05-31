import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from cutie.dataset.compressed_davis_dataset import CompressedDAVISDataset


def tensor_to_image(rgb: torch.Tensor) -> Image.Image:
    array = (rgb.permute(1, 2, 0).numpy().clip(0, 1) * 255).astype(np.uint8)
    return Image.fromarray(array)


def overlay_mask(image: Image.Image, mask: np.ndarray) -> Image.Image:
    base = image.convert('RGBA')
    overlay = Image.new('RGBA', base.size, (0, 0, 0, 0))
    mask_bool = mask > 0
    if mask_bool.any():
        alpha = (mask_bool.astype(np.uint8) * 120)
        color = np.zeros((*mask.shape, 4), dtype=np.uint8)
        color[..., 0] = 255
        color[..., 3] = alpha
        overlay = Image.fromarray(color, mode='RGBA')
    return Image.alpha_composite(base, overlay).convert('RGB')


def draw_mv(image: Image.Image, mv: np.ndarray, stride: int = 48, scale: float = 1.0) -> Image.Image:
    out = image.convert('RGB')
    draw = ImageDraw.Draw(out)
    h, w = mv.shape[:2]
    for y in range(stride // 2, h, stride):
        for x in range(stride // 2, w, stride):
            dx, dy = mv[y, x]
            mag = float(np.hypot(dx, dy))
            if mag < 0.5:
                continue
            x2 = x + dx * scale
            y2 = y + dy * scale
            draw.line((x, y, x2, y2), fill=(255, 220, 0), width=2)
            draw.ellipse((x2 - 2, y2 - 2, x2 + 2, y2 + 2), fill=(255, 80, 0))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--davis-root', type=Path, default=Path('data/DAVIS/2017/trainval'))
    parser.add_argument('--compressed-root', type=Path, default=Path('data/DAVIS/2017/trainval/compressed/3M-GOP12'))
    parser.add_argument('--output-dir', type=Path, default=Path('output/sanity/creff_480_l8_batch'))
    parser.add_argument('--size', type=int, default=480)
    parser.add_argument('--resize-mode', choices=['crop', 'square'], default='crop')
    parser.add_argument('--seq-length', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-objects', type=int, default=3)
    parser.add_argument('--seed', type=int, default=14159265)
    parser.add_argument('--mv-stride', type=int, default=48)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dataset = CompressedDAVISDataset(args.davis_root,
                                     args.compressed_root,
                                     seq_length=args.seq_length,
                                     size=args.size,
                                     max_num_obj=args.num_objects,
                                     seed=args.seed,
                                     resize_mode=args.resize_mode)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    batch = next(iter(loader))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stats_lines = []
    for bi in range(args.batch_size):
        name = batch['info']['name'][bi]
        sample_dir = args.output_dir / f'sample_{bi}_{name}'
        sample_dir.mkdir(parents=True, exist_ok=True)
        rgb = batch['rgb'][bi]
        cls_gt = batch['cls_gt'][bi, :, 0].numpy()
        mv = batch['mv'][bi].numpy()
        frames = batch['info']['frames']
        frame_names = [frames[ti][bi] for ti in range(args.seq_length)]
        stats_lines.append(f'sample {bi} name={name} frames={frame_names}')
        for ti in range(args.seq_length):
            image = tensor_to_image(rgb[ti].cpu())
            mask_overlay = overlay_mask(image, cls_gt[ti])
            mask_overlay.save(sample_dir / f'{ti:02d}_{frame_names[ti]}_rgb_mask.png')
            mv_image = draw_mv(image, mv[ti], stride=args.mv_stride)
            mv_image.save(sample_dir / f'{ti:02d}_{frame_names[ti]}_mv.png')

            mag = np.linalg.norm(mv[ti], axis=-1)
            nz = mag[mag > 0]
            values = nz if nz.size else mag.reshape(-1)
            stats_lines.append(
                f'  t={ti:02d} frame={frame_names[ti]} '
                f'mv_abs_mean={float(values.mean()):.3f} '
                f'p50={float(np.percentile(values, 50)):.3f} '
                f'p95={float(np.percentile(values, 95)):.3f} '
                f'p99={float(np.percentile(values, 99)):.3f} '
                f'max={float(values.max()):.3f}')

    stats_path = args.output_dir / 'mv_stats.txt'
    stats_path.write_text('\n'.join(stats_lines) + '\n')
    print(f'wrote {args.output_dir}')
    print(stats_path.read_text())


if __name__ == '__main__':
    main()
