import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from cutie.dataset.compressed_davis_dataset import CompressedDAVISDataset
from cutie.model.losses import LossComputer
from cutie.model.train_wrapper import CutieTrainWrapper


def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % (2**31) + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def next_batch(data_iter, loader):
    try:
        return next(data_iter), data_iter
    except StopIteration:
        data_iter = iter(loader)
        return next(data_iter), data_iter


def move_batch_to_device(data, device):
    for key, value in list(data.items()):
        if isinstance(value, torch.Tensor):
            data[key] = value.to(device, non_blocking=True)
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--davis-root', type=Path, default=Path('data/DAVIS/2017/trainval'))
    parser.add_argument('--compressed-root',
                        type=Path,
                        default=Path('data/DAVIS/2017/trainval/compressed/3M-GOP12'))
    parser.add_argument('--weights', type=Path, default=Path('weights/cutie-base-mega.pth'))
    parser.add_argument('--resume', type=Path, default=None)
    parser.add_argument('--start-step', type=int, default=0)
    parser.add_argument('--steps', type=int, default=1500)
    parser.add_argument('--size', type=int, default=240)
    parser.add_argument('--resize-mode', choices=['crop', 'square'], default='crop')
    parser.add_argument('--seq-length', type=int, default=8)
    parser.add_argument('--num-objects', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--reference-batch-size', type=int, default=32)
    parser.add_argument('--no-scale-lr', action='store_true')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--grad-accum', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--prefetch-factor', type=int, default=4)
    parser.add_argument('--pin-memory', action='store_true')
    parser.add_argument('--persistent-workers', action='store_true')
    parser.add_argument('--save-dir', type=Path, default=Path('output/stage1_plan34_lr_adapt_s240_l8_b4a8'))
    parser.add_argument('--save-every', type=int, default=500)
    parser.add_argument('--seed', type=int, default=14159265)
    parser.add_argument('--amp', action='store_true')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    dataset = CompressedDAVISDataset(args.davis_root,
                                     args.compressed_root,
                                     seq_length=args.seq_length,
                                     size=args.size,
                                     max_num_obj=args.num_objects,
                                     seed=args.seed,
                                     resize_mode=args.resize_mode)
    loader = DataLoader(dataset,
                        batch_size=args.batch_size,
                        shuffle=True,
                        num_workers=args.num_workers,
                        pin_memory=args.pin_memory,
                        drop_last=True,
                        persistent_workers=args.persistent_workers and args.num_workers > 0,
                        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
                        worker_init_fn=worker_init_fn)
    print(f'Found {len(dataset)} trainable GOP-start clips.', flush=True)
    print(f'DataLoader batches per epoch: {len(loader)}.', flush=True)
    effective_batch_size = args.batch_size * args.grad_accum
    train_lr = args.lr if args.no_scale_lr else args.lr * effective_batch_size / args.reference_batch_size
    print('Training geometry:',
          f'size={args.size}',
          f'resize_mode={args.resize_mode}',
          f'seq_length={args.seq_length}',
          f'batch_size={args.batch_size}',
          f'grad_accum={args.grad_accum}',
          f'effective_batch_size={effective_batch_size}',
          f'lr={train_lr}',
          'stage=stage1_plan34',
          'trainable=pixel_encoder.layer2+layer3',
          'creff=disabled',
          'feat_distill=disabled',
          flush=True)

    model_cfg = OmegaConf.load('cutie/config/model/base.yaml')
    cfg = OmegaConf.create({
        'model': model_cfg,
        'use_creff': False,
        'debug': True,
    })
    cfg.model.use_creff = False

    stage_cfg = OmegaConf.create({
        'name': 'compressed_dataloader',
        'num_objects': args.num_objects,
        'seq_length': args.seq_length,
        'num_ref_frames': 2,
        'deep_update_prob': 0.0,
        'amp': args.amp,
        'lr_scale': 1.0,
        'freeze_decoder_for_fst': False,
        'feat_distill_weight': 0.0,
        'point_supervision': True,
        'train_num_points': 2048,
        'oversample_ratio': 3.0,
        'importance_sample_ratio': 0.75,
    })

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CutieTrainWrapper(cfg, stage_cfg).to(device).train()
    weights = torch.load(args.weights, map_location='cpu')
    model.load_weights(weights)
    if args.resume is not None:
        model.load_state_dict(torch.load(args.resume, map_location='cpu'))
        print(f'Resumed model state from {args.resume} at start_step={args.start_step}.', flush=True)

    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if name.startswith(('pixel_encoder.layer2.', 'pixel_encoder.layer3.')):
            param.requires_grad = True

    trainable = [(name, param.numel()) for name, param in model.named_parameters()
                 if param.requires_grad]
    print(f'Trainable parameter tensors: {len(trainable)}.', flush=True)
    print(f'Trainable parameters: {sum(n for _, n in trainable)}.', flush=True)
    print('Trainable names: ' + ', '.join(name for name, _ in trainable), flush=True)

    loss_computer = LossComputer(cfg, stage_cfg)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=train_lr,
                                  weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == 'cuda')
    args.save_dir.mkdir(parents=True, exist_ok=True)

    data_iter = iter(loader)
    optimizer.zero_grad(set_to_none=True)
    for step in range(args.start_step, args.steps):
        accum_losses = {}
        first_name = None
        for accum_idx in range(args.grad_accum):
            data, data_iter = next_batch(data_iter, loader)
            data = move_batch_to_device(data, device)

            out = model(data)
            losses = loss_computer.compute({**data, **out}, out['num_filled_objects'])
            scaled_loss = losses['total_loss'] / args.grad_accum
            scaler.scale(scaled_loss).backward()

            for key, value in losses.items():
                accum_losses[key] = accum_losses.get(key, 0.0) + float(value.detach().cpu()) / args.grad_accum
            if first_name is None:
                names = data['info']['name']
                first_name = names[0] if isinstance(names, list) else names

        if step == args.start_step:
            layer3_grad = model.pixel_encoder.layer3[-1].conv1.weight.grad
            layer1_grad = model.pixel_encoder.res2[-1].conv1.weight.grad
            decoder_grad = next(model.mask_decoder.parameters()).grad
            print('Freeze sanity:',
                  f'layer3_grad={layer3_grad is not None}',
                  f'layer1_grad={layer1_grad is not None}',
                  f'mask_decoder_grad={decoder_grad is not None}', flush=True)

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        print('step', step,
              'stage', 'stage1_plan34',
              'batch_size', args.batch_size,
              'grad_accum', args.grad_accum,
              'effective_batch_size', effective_batch_size,
              'clip', first_name,
              'total_loss', accum_losses.get('total_loss', 0.0),
              'loss_ce', accum_losses.get('loss_ce', 0.0),
              'loss_dice', accum_losses.get('loss_dice', 0.0),
              'feat_distill', accum_losses.get('feat_distill', 0.0),
              'grad_norm', float(grad_norm.detach().cpu()), flush=True)

        if (step + 1) % args.save_every == 0 or step + 1 == args.steps:
            ckpt_path = args.save_dir / f'stage1_plan34_step_{step + 1}.pth'
            torch.save(model.state_dict(), ckpt_path)
            print(f'saved {ckpt_path}', flush=True)


if __name__ == '__main__':
    main()
