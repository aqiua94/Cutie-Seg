import argparse
import csv
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def run(cmd: list[str]) -> None:
    print('RUN ' + ' '.join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def read_global(metrics_file: Path) -> dict[str, str]:
    with metrics_file.open(newline='') as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        if row['video'] == 'global':
            return row
    raise RuntimeError(f'No global row in {metrics_file}')


def append_csv(path: Path, row: dict[str, str]) -> None:
    exists = path.exists()
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
    if exists and row['checkpoint'] in path.read_text() and row['metrics_file'] in path.read_text():
        print(f'CSV ledger already has {row["checkpoint"]}', flush=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def append_md(path: Path, row: dict[str, str], eval_params_short: str) -> None:
    line = (
        f'| {row["validation_time_utc"][:16]} | {row["method"]} | '
        f'`{row["checkpoint"]}` | {row["train_step"]} | DAVIS-2017 val | '
        f'{eval_params_short} | {float(row["J"]):.4f} | {float(row["F"]):.4f} | '
        f'{float(row["J_and_F"]):.4f} | {row["frames"]} | '
        f'`{row["metrics_file"]}` |'
    )
    if path.exists():
        text = path.read_text()
        if row['checkpoint'] in text and row['metrics_file'] in text:
            print(f'MD ledger already has {row["checkpoint"]}', flush=True)
            return
        marker = '\n## Notes\n'
        if marker in text:
            path.write_text(text.replace(marker, line + '\n' + marker, 1))
            return
        path.write_text(text.rstrip() + '\n' + line + '\n')
    else:
        path.write_text('# Validation Ledger\n\n' + line + '\n')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--method', default='CReFF-only')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--ledger-csv', type=Path, default=Path('output/validation_ledger.csv'))
    parser.add_argument('--ledger-md', type=Path, default=Path('output/validation_ledger.md'))
    parser.add_argument('--size', type=int, default=480)
    parser.add_argument('--gop-length', type=int, default=12)
    parser.add_argument('--creff-k', type=int, default=7)
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--poll-seconds', type=int, default=60)
    parser.add_argument('--timeout-seconds', type=int, default=24 * 3600)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--disable-creff', action='store_true')
    parser.add_argument('--compressed-p-frames', action='store_true')
    parser.add_argument('--lr-scale', type=float, default=0.5)
    parser.add_argument('--train-lr-desc', default='lr_creff=1e-4')
    parser.add_argument('--train-batch-size', type=int, default=4)
    parser.add_argument('--train-grad-accum', type=int, default=8)
    parser.add_argument('--train-seq-length', type=int, default=8)
    parser.add_argument('--train-size-desc', default='480x480_square_crop')
    parser.add_argument('--trainable-desc', default='CReFF only')
    parser.add_argument('--frozen-desc', default='frozen encoder/decoder')
    parser.add_argument('--notes', default='auto-evaluated after 480/seq8 CReFF-only training checkpoint')
    args = parser.parse_args()

    start = time.time()
    while not args.checkpoint.exists():
        elapsed = time.time() - start
        if elapsed > args.timeout_seconds:
            raise TimeoutError(f'Timed out waiting for {args.checkpoint}')
        print(f'waiting for {args.checkpoint} elapsed={elapsed:.0f}s', flush=True)
        time.sleep(args.poll_seconds)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    eval_cmd = [
        sys.executable,
        'scripts/eval_compressed_davis.py',
        '--weights',
        str(args.checkpoint),
        '--output-dir',
        str(args.output_dir),
        '--size',
        str(args.size),
        '--gop-length',
        str(args.gop_length),
        '--creff-k',
        str(args.creff_k),
    ]
    if args.amp:
        eval_cmd.append('--amp')
    if args.disable_creff:
        eval_cmd.append('--disable-creff')
    if args.compressed_p_frames:
        eval_cmd.append('--compressed-p-frames')
    eval_cmd.extend(['--lr-scale', str(args.lr_scale)])
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

    global_row = read_global(metrics_file)
    now = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    eff_bs = args.train_batch_size * args.train_grad_accum
    train_params = (
        f'train_size={args.train_size_desc}; batch_size={args.train_batch_size}; '
        f'grad_accum={args.train_grad_accum}; eff_bs={eff_bs}; '
        f'seq_length={args.train_seq_length}; creff_k={args.creff_k}; trainable={args.trainable_desc}; '
        f'{args.frozen_desc}; compressed DAVIS; GOP12; {args.train_lr_desc}; no AMP'
    )
    eval_params = (
        f'compressed_root=data/DAVIS/2017/trainval/compressed/3M-GOP12; '
        f'gop_length={args.gop_length}; creff_k={args.creff_k}; size={args.size}; '
        f'{"disable_creff; " if args.disable_creff else ""}'
        f'{"compressed_p_frames; " if args.compressed_p_frames else ""}'
        f'lr_scale={args.lr_scale}; '
        f'{"amp; " if args.amp else ""}skip_first; eval_size={args.size}; '
        f'jf_workers={args.workers}'
    )
    row = {
        'validation_time_utc': now,
        'method': args.method,
        'checkpoint': str(args.checkpoint),
        'train_step': str(args.step),
        'train_params': train_params,
        'eval_dataset': 'DAVIS-2017 val',
        'eval_subset': 'full val',
        'eval_params': eval_params,
        'metric_scope': 'full-val J&F',
        'J': global_row['J'],
        'F': global_row['F'],
        'J_and_F': global_row['J&F'],
        'frames': global_row['frames'],
        'pred_dir': str(args.output_dir / 'Annotations'),
        'metrics_file': str(metrics_file),
        'notes': args.notes,
    }
    append_csv(args.ledger_csv, row)
    append_md(
        args.ledger_md,
        row,
        f'train 480 crop, seq8, GOP12, k{args.creff_k}, size {args.size}, '
        f'{"AMP, " if args.amp else ""}skip first, eval-size {args.size}, J&F workers {args.workers}',
    )
    print(f'appended validation ledger: J={row["J"]} F={row["F"]} J&F={row["J_and_F"]}', flush=True)


if __name__ == '__main__':
    main()
