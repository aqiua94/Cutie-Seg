import argparse
import re
import subprocess
import sys
import time
from pathlib import Path


STEP_RE = re.compile(r'compressed_davis_stage1_encoder_step_(\d+)\.pth$')


def latest_checkpoint(save_dir: Path) -> tuple[int, Path | None]:
    best_step = 0
    best_path = None
    for path in save_dir.glob('compressed_davis_stage1_encoder_step_*.pth'):
        match = STEP_RE.search(path.name)
        if not match:
            continue
        step = int(match.group(1))
        if step > best_step:
            best_step = step
            best_path = path
    return best_step, best_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--save-dir', type=Path, required=True)
    parser.add_argument('--target-step', type=int, default=1500)
    parser.add_argument('--size', type=int, default=480)
    parser.add_argument('--resize-mode', default='crop')
    parser.add_argument('--seq-length', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--grad-accum', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--prefetch-factor', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--save-every', type=int, default=100)
    parser.add_argument('--i-anchor-weight', type=float, default=0.0)
    parser.add_argument('--distill-loss', choices=['cosine', 'mse'], default='cosine')
    parser.add_argument('--sleep-after-exit', type=int, default=15)
    args = parser.parse_args()

    while True:
        step, ckpt = latest_checkpoint(args.save_dir)
        print(f'latest checkpoint step={step} path={ckpt}', flush=True)
        if step >= args.target_step:
            print(f'target reached: {step} >= {args.target_step}', flush=True)
            return

        cmd = [
            sys.executable,
            'scripts/train_compressed_davis_stage1_encoder.py',
            '--weights', str(args.weights),
            '--steps', str(args.target_step),
            '--size', str(args.size),
            '--resize-mode', args.resize_mode,
            '--seq-length', str(args.seq_length),
            '--batch-size', str(args.batch_size),
            '--grad-accum', str(args.grad_accum),
            '--num-workers', str(args.num_workers),
            '--prefetch-factor', str(args.prefetch_factor),
            '--lr', str(args.lr),
            '--save-dir', str(args.save_dir),
            '--save-every', str(args.save_every),
            '--i-anchor-weight', str(args.i_anchor_weight),
            '--distill-loss', args.distill_loss,
        ]
        if ckpt is not None:
            cmd.extend(['--resume', str(ckpt), '--start-step', str(step)])

        print('RUN ' + ' '.join(cmd), flush=True)
        result = subprocess.run(cmd)
        print(f'train subprocess exited with code {result.returncode}', flush=True)
        time.sleep(args.sleep_after_exit)


if __name__ == '__main__':
    main()
