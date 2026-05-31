import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image


def load_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path), dtype=np.int64)


def resize_short(mask: np.ndarray, size: int) -> np.ndarray:
    if size <= 0 or min(mask.shape[:2]) == size:
        return mask
    h, w = mask.shape[:2]
    new_h = int(h / min(h, w) * size)
    new_w = int(w / min(h, w) * size)
    return np.asarray(
        Image.fromarray(mask.astype(np.uint8)).resize((new_w, new_h), Image.NEAREST),
        dtype=np.int64,
    )


def binary_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0
    return float(inter / union)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--gt-root',
        type=Path,
        default=Path('data/DAVIS/2017/trainval/Annotations/Full-Resolution'),
    )
    parser.add_argument('--pred-root', type=Path, required=True)
    parser.add_argument(
        '--subset',
        type=Path,
        default=Path('data/DAVIS/2017/trainval/ImageSets/2017/val.txt'),
    )
    parser.add_argument('--output-csv', type=Path, default=None)
    parser.add_argument('--skip-first', action='store_true')
    parser.add_argument('--eval-size', type=int, default=-1)
    args = parser.parse_args()

    videos = [line.strip() for line in args.subset.read_text().splitlines() if line.strip()]
    rows = []
    all_j = []
    missing = []

    for video in videos:
        gt_dir = args.gt_root / video
        pred_dir = args.pred_root / video
        frame_names = sorted(p.name for p in gt_dir.glob('*.png'))
        if args.skip_first:
            frame_names = frame_names[1:]

        video_j = []
        for frame_name in frame_names:
            gt_path = gt_dir / frame_name
            pred_path = pred_dir / frame_name
            if not pred_path.exists():
                missing.append(str(pred_path))
                continue

            gt = load_mask(gt_path)
            pred = load_mask(pred_path)
            if pred.shape != gt.shape:
                pred = np.asarray(
                    Image.fromarray(pred.astype(np.uint8)).resize((gt.shape[1], gt.shape[0]),
                                                                  Image.NEAREST),
                    dtype=np.int64,
                )
            if args.eval_size > 0:
                gt = resize_short(gt, args.eval_size)
                pred = resize_short(pred, args.eval_size)

            labels = np.unique(gt)
            labels = labels[labels != 0]
            for label in labels:
                j = binary_iou(pred == label, gt == label)
                video_j.append(j)
                all_j.append(j)

        rows.append({
            'video': video,
            'J': float(np.mean(video_j)) if video_j else float('nan'),
            'frames': len(frame_names),
        })

    rows.append({
        'video': 'global',
        'J': float(np.mean(all_j)) if all_j else float('nan'),
        'frames': sum(r['frames'] for r in rows),
    })

    for row in rows:
        print(f"{row['video']:>24s}  J {row['J']:.4f}  frames {row['frames']}")

    if missing:
        print(f'missing predictions: {len(missing)}')
        for item in missing[:20]:
            print(f'  {item}')

    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['video', 'J', 'frames'])
            writer.writeheader()
            writer.writerows(rows)


if __name__ == '__main__':
    main()
