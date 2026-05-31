import argparse
import csv
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def run(cmd: list[str]) -> None:
    print('RUN ' + ' '.join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def global_metrics(path: Path) -> dict[str, str]:
    with path.open(newline='') as f:
        for row in csv.DictReader(f):
            if row['video'] == 'global':
                return row
    raise RuntimeError(f'No global row in {path}')


def append_csv(path: Path, row: dict[str, str]) -> None:
    fieldnames = [
        'validation_time_utc',
        'method',
        'checkpoint',
        'train_step',
        'train_params',
        'eval_dataset',
        'eval_subset',
        'eval_params',
        'metric_scope',
        'J',
        'F',
        'J_and_F',
        'frames',
        'pred_dir',
        'metrics_file',
        'notes',
    ]
    if path.exists() and row['method'] in path.read_text() and row['metrics_file'] in path.read_text():
        print(f'CSV ledger already has {row["method"]}', flush=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if path.stat().st_size == 0:
            writer.writeheader()
        writer.writerow(row)


def append_md(path: Path, row: dict[str, str], short_params: str) -> None:
    line = (
        f'| {row["validation_time_utc"][:16]} | {row["method"]} | '
        f'`{row["checkpoint"]}` | {row["train_step"]} | DAVIS-2017 val | '
        f'{short_params} | {float(row["J"]):.4f} | {float(row["F"]):.4f} | '
        f'{float(row["J_and_F"]):.4f} | {row["frames"]} | '
        f'`{row["metrics_file"]}` |'
    )
    text = path.read_text() if path.exists() else '# Validation Ledger\n\n## Current Results\n\n'
    if row['method'] in text and row['metrics_file'] in text:
        print(f'MD ledger already has {row["method"]}', flush=True)
        return
    marker = '\n## Notes\n'
    if marker in text:
        text = text.replace(marker, line + '\n' + marker, 1)
    else:
        text = text.rstrip() + '\n' + line + '\n'
    path.write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=Path, default=Path('weights/cutie-base-mega.pth'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--method', required=True)
    parser.add_argument('--gop-length', type=int, required=True)
    parser.add_argument('--size', type=int, default=480)
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--compressed-p-frames', action='store_true',
                        help='Use LR P-frame encoding while keeping CReFF disabled.')
    parser.add_argument('--ledger-csv', type=Path, default=Path('output/validation_ledger.csv'))
    parser.add_argument('--ledger-md', type=Path, default=Path('output/validation_ledger.md'))
    args = parser.parse_args()

    eval_cmd = [
        sys.executable,
        'scripts/eval_compressed_davis.py',
        '--weights',
        str(args.weights),
        '--output-dir',
        str(args.output_dir),
        '--size',
        str(args.size),
        '--gop-length',
        str(args.gop_length),
        '--disable-creff',
    ]
    if args.amp:
        eval_cmd.append('--amp')
    if args.compressed_p_frames:
        eval_cmd.append('--compressed-p-frames')
    run(eval_cmd)

    metrics_file = args.output_dir / f'jf_metrics_{args.size}_parallel.csv'
    run([
        sys.executable,
        'scripts/evaluate_davis_jf_parallel.py',
        '--pred-root',
        str(args.output_dir / 'Annotations'),
        '--skip-first',
        '--eval-size',
        str(args.size),
        '--workers',
        str(args.workers),
        '--output-csv',
        str(metrics_file),
    ])

    metrics = global_metrics(metrics_file)
    now = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    eval_params = (
        f'compressed_root=data/DAVIS/2017/trainval/compressed/3M-GOP12; '
        f'gop_length={args.gop_length}; disable_creff; '
        f'{"compressed_p_frames; " if args.compressed_p_frames else ""}'
        f'size={args.size}; '
        f'{"amp; " if args.amp else ""}skip_first; eval_size={args.size}; '
        f'jf_workers={args.workers}'
    )
    row = {
        'validation_time_utc': now,
        'method': args.method,
        'checkpoint': str(args.weights),
        'train_step': '0',
        'train_params': 'official cutie-base-mega; no CReFF; no training',
        'eval_dataset': 'DAVIS-2017 val',
        'eval_subset': 'full val',
        'eval_params': eval_params,
        'metric_scope': 'full-val J&F',
        'J': metrics['J'],
        'F': metrics['F'],
        'J_and_F': metrics['J&F'],
        'frames': metrics['frames'],
        'pred_dir': str(args.output_dir / 'Annotations'),
        'metrics_file': str(metrics_file),
        'notes': ('true lower bound: LR P-frame encode without CReFF injection'
                  if args.compressed_p_frames else
                  'reference run on HEVC decoded frames with CReFF disabled'),
    }
    append_csv(args.ledger_csv, row)
    append_md(
        args.ledger_md,
        row,
        f'GOP{args.gop_length}, no-CReFF, '
        f'{"LR P-frames, " if args.compressed_p_frames else ""}'
        f'size {args.size}, '
        f'{"AMP, " if args.amp else ""}skip first, eval-size {args.size}, J&F workers {args.workers}',
    )
    print(f'{args.method}: J={row["J"]} F={row["F"]} J&F={row["J_and_F"]}', flush=True)


if __name__ == '__main__':
    main()
