import argparse
import csv
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

try:
    import cv2
except ImportError:
    cv2 = None


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


def binary_boundary(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    mask_u8 = mask.astype(np.uint8)
    if cv2 is not None:
        eroded = cv2.erode(mask_u8, np.ones((3, 3), np.uint8), iterations=1).astype(bool)
    else:
        image = Image.fromarray(mask_u8 * 255)
        eroded = np.asarray(image.filter(ImageFilter.MinFilter(3)), dtype=np.uint8) > 0
    return mask.astype(bool) ^ eroded


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool)
    mask_u8 = mask.astype(np.uint8)
    if cv2 is not None:
        kernel = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
        return cv2.dilate(mask_u8, kernel, iterations=1).astype(bool)
    image = Image.fromarray(mask_u8 * 255)
    out = image.filter(ImageFilter.MaxFilter(2 * radius + 1))
    return np.asarray(out, dtype=np.uint8) > 0


def boundary_f(pred: np.ndarray, gt: np.ndarray, bound_th: float = 0.008) -> float:
    pred_b = binary_boundary(pred)
    gt_b = binary_boundary(gt)
    if not pred_b.any() and not gt_b.any():
        return 1.0
    if not pred_b.any() or not gt_b.any():
        return 0.0

    h, w = gt.shape
    radius = max(1, int(np.ceil(bound_th * np.linalg.norm([h, w]))))
    pred_match = np.logical_and(pred_b, dilate(gt_b, radius)).sum()
    gt_match = np.logical_and(gt_b, dilate(pred_b, radius)).sum()
    precision = pred_match / max(pred_b.sum(), 1)
    recall = gt_match / max(gt_b.sum(), 1)
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def evaluate_video(task):
    video, gt_root, pred_root, skip_first, eval_size = task
    gt_dir = gt_root / video
    pred_dir = pred_root / video
    frame_names = sorted(p.name for p in gt_dir.glob('*.png'))
    if skip_first:
        frame_names = frame_names[1:]

    video_j = []
    video_f = []
    missing = []

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
        if eval_size > 0:
            gt = resize_short(gt, eval_size)
            pred = resize_short(pred, eval_size)

        labels = np.unique(gt)
        labels = labels[labels != 0]
        for label in labels:
            gt_obj = gt == label
            pred_obj = pred == label
            video_j.append(binary_iou(pred_obj, gt_obj))
            video_f.append(boundary_f(pred_obj, gt_obj))

    return {
        'video': video,
        'J': float(np.mean(video_j)) if video_j else float('nan'),
        'F': float(np.mean(video_f)) if video_f else float('nan'),
        'J&F': float((np.mean(video_j) + np.mean(video_f)) / 2) if video_j else float('nan'),
        'frames': len(frame_names),
        'all_j': video_j,
        'all_f': video_f,
        'missing': missing,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt-root',
                        type=Path,
                        default=Path('data/DAVIS/2017/trainval/Annotations/Full-Resolution'))
    parser.add_argument('--pred-root', type=Path, required=True)
    parser.add_argument('--subset',
                        type=Path,
                        default=Path('data/DAVIS/2017/trainval/ImageSets/2017/val.txt'))
    parser.add_argument('--output-csv', type=Path, default=None)
    parser.add_argument('--skip-first', action='store_true')
    parser.add_argument('--eval-size', type=int, default=-1)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()

    videos = [line.strip() for line in args.subset.read_text().splitlines() if line.strip()]
    tasks = [(video, args.gt_root, args.pred_root, args.skip_first, args.eval_size)
             for video in videos]

    if args.workers <= 1:
        results = [evaluate_video(task) for task in tasks]
    else:
        with Pool(processes=args.workers) as pool:
            results = list(pool.imap(evaluate_video, tasks))

    rows = []
    all_j = []
    all_f = []
    missing = []
    for result in results:
        rows.append({
            'video': result['video'],
            'J': result['J'],
            'F': result['F'],
            'J&F': result['J&F'],
            'frames': result['frames'],
        })
        all_j.extend(result['all_j'])
        all_f.extend(result['all_f'])
        missing.extend(result['missing'])

    rows.append({
        'video': 'global',
        'J': float(np.mean(all_j)) if all_j else float('nan'),
        'F': float(np.mean(all_f)) if all_f else float('nan'),
        'J&F': float((np.mean(all_j) + np.mean(all_f)) / 2) if all_j else float('nan'),
        'frames': sum(r['frames'] for r in rows),
    })

    for row in rows:
        print(f"{row['video']:>24s}  J {row['J']:.4f}  F {row['F']:.4f}  J&F {row['J&F']:.4f}  frames {row['frames']}",
              flush=True)

    if missing:
        print(f'missing predictions: {len(missing)}', flush=True)
        for item in missing[:20]:
            print(f'  {item}', flush=True)

    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['video', 'J', 'F', 'J&F', 'frames'])
            writer.writeheader()
            writer.writerows(rows)


if __name__ == '__main__':
    main()
