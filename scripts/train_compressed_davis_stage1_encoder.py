import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from cutie.dataset.compressed_davis_dataset import CompressedDAVISDataset
from cutie.model.cutie import CUTIE


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


def build_model(weights: Path, device: torch.device) -> CUTIE:
    model_cfg = OmegaConf.load('cutie/config/model/base.yaml')
    cfg = OmegaConf.create({'model': model_cfg, 'use_creff': True, 'creff_k': 7, 'debug': True})
    cfg.model.use_creff = True
    model = CUTIE(cfg).to(device)
    state = torch.load(weights, map_location='cpu')
    model.load_state_dict(state)
    return model


def freeze_for_stage1(model: CUTIE) -> list[str]:
    for param in model.parameters():
        param.requires_grad = False
    trainable_prefixes = ('pixel_encoder.layer2.', 'pixel_encoder.layer3.')
    trainable_names = []
    for name, param in model.named_parameters():
        if name.startswith(trainable_prefixes):
            param.requires_grad = True
            trainable_names.append(name)
    return trainable_names


def feature_loss(student: torch.Tensor, teacher: torch.Tensor, loss_type: str) -> torch.Tensor:
    if student.shape[-2:] != teacher.shape[-2:]:
        teacher = F.interpolate(teacher, size=student.shape[-2:], mode='bilinear', align_corners=False)
    student = student.float()
    teacher = teacher.float()
    if loss_type == 'cosine':
        student = F.normalize(student, dim=1)
        teacher = F.normalize(teacher, dim=1)
        return 1.0 - (student * teacher).sum(dim=1).mean()
    if loss_type == 'mse':
        return F.mse_loss(student, teacher)
    raise ValueError(f'Unknown loss_type: {loss_type}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--davis-root', type=Path, default=Path('data/DAVIS/2017/trainval'))
    parser.add_argument('--compressed-root', type=Path, default=Path('data/DAVIS/2017/trainval/compressed/3M-GOP12'))
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--resume', type=Path, default=None)
    parser.add_argument('--start-step', type=int, default=0)
    parser.add_argument('--steps', type=int, default=1500)
    parser.add_argument('--size', type=int, default=480)
    parser.add_argument('--resize-mode', choices=['crop', 'square'], default='crop')
    parser.add_argument('--seq-length', type=int, default=8)
    parser.add_argument('--num-objects', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--grad-accum', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--prefetch-factor', type=int, default=1)
    parser.add_argument('--pin-memory', action='store_true')
    parser.add_argument('--persistent-workers', action='store_true')
    parser.add_argument('--lr-scale', type=float, default=0.5)
    parser.add_argument('--pix-weight', type=float, default=1.0)
    parser.add_argument('--ms-weight', type=float, default=0.25)
    parser.add_argument('--i-anchor-weight', type=float, default=0.0)
    parser.add_argument('--distill-loss', choices=['cosine', 'mse'], default='cosine')
    parser.add_argument('--save-dir', type=Path, default=Path('output/creff_davis_stage1_encoder_s480_l8_b2a16'))
    parser.add_argument('--save-every', type=int, default=100)
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
    effective_batch_size = args.batch_size * args.grad_accum
    print(f'Found {len(dataset)} trainable GOP-start clips.', flush=True)
    print(f'DataLoader batches per epoch: {len(loader)}.', flush=True)
    print('Stage1 geometry:',
          f'size={args.size}',
          f'resize_mode={args.resize_mode}',
          f'seq_length={args.seq_length}',
          f'batch_size={args.batch_size}',
          f'grad_accum={args.grad_accum}',
          f'effective_batch_size={effective_batch_size}',
          f'lr={args.lr}',
          f'lr_scale={args.lr_scale}',
          f'distill_loss={args.distill_loss}',
          f'i_anchor_weight={args.i_anchor_weight}',
          flush=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    student = build_model(args.weights, device).train()
    teacher = build_model(args.weights, device).eval()
    for param in teacher.parameters():
        param.requires_grad = False

    if args.resume is not None:
        student.load_state_dict(torch.load(args.resume, map_location='cpu'))
        print(f'Resumed student state from {args.resume} at start_step={args.start_step}.', flush=True)

    trainable_names = freeze_for_stage1(student)
    trainable = [(name, param.numel()) for name, param in student.named_parameters() if param.requires_grad]
    print(f'Trainable parameter tensors: {len(trainable)}.', flush=True)
    print(f'Trainable parameters: {sum(n for _, n in trainable)}.', flush=True)
    print('Trainable names: ' + ', '.join(trainable_names), flush=True)

    optimizer = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad],
                                  lr=args.lr,
                                  weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           T_max=max(1, args.steps - args.start_step),
                                                           eta_min=args.lr * 0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == 'cuda')
    args.save_dir.mkdir(parents=True, exist_ok=True)

    data_iter = iter(loader)
    optimizer.zero_grad(set_to_none=True)
    for step in range(args.start_step, args.steps):
        accum = {'total_loss': 0.0, 'p_pix': 0.0, 'p_ms': 0.0, 'i_anchor': 0.0}
        first_name = None
        for _ in range(args.grad_accum):
            data, data_iter = next_batch(data_iter, loader)
            data = move_batch_to_device(data, device)
            frames = data['rgb']
            is_i_frame = data['is_i_frame'].bool()
            b, t = frames.shape[:2]
            frames_flat = frames.flatten(0, 1)
            i_frame_flags = is_i_frame.clone()
            i_frame_flags[:, 0] = True
            p_mask = ~i_frame_flags.flatten(0, 1)
            i_mask = i_frame_flags.flatten(0, 1)
            if not p_mask.any():
                p_mask = torch.ones((b * t,), device=device, dtype=torch.bool)
            frames_p = frames_flat[p_mask]
            image_lr = F.interpolate(frames_p,
                                     scale_factor=args.lr_scale,
                                     mode='bilinear',
                                     align_corners=False)

            with torch.cuda.amp.autocast(enabled=args.amp):
                student_ms, student_pix = student.encode_image(image_lr)
                with torch.no_grad():
                    teacher_ms, teacher_pix = teacher.encode_image(frames_p)
                p_pix = feature_loss(student_pix, teacher_pix, args.distill_loss)
                p_ms_losses = [feature_loss(s, h, args.distill_loss) for s, h in zip(student_ms, teacher_ms)]
                p_ms = torch.stack(p_ms_losses).mean()
                p_loss = args.pix_weight * p_pix + args.ms_weight * p_ms

                if args.i_anchor_weight > 0 and i_mask.any():
                    frames_i = frames_flat[i_mask]
                    student_i_ms, student_i_pix = student.encode_image(frames_i)
                    with torch.no_grad():
                        teacher_i_ms, teacher_i_pix = teacher.encode_image(frames_i)
                    i_pix = feature_loss(student_i_pix, teacher_i_pix, args.distill_loss)
                    i_ms_losses = [feature_loss(s, h, args.distill_loss)
                                   for s, h in zip(student_i_ms, teacher_i_ms)]
                    i_anchor = i_pix + args.ms_weight * torch.stack(i_ms_losses).mean()
                else:
                    i_anchor = frames.new_tensor(0.0)

                total_loss = p_loss + args.i_anchor_weight * i_anchor
                scaled_loss = total_loss / args.grad_accum
            scaler.scale(scaled_loss).backward()

            accum['total_loss'] += float(total_loss.detach().cpu()) / args.grad_accum
            accum['p_pix'] += float(p_pix.detach().cpu()) / args.grad_accum
            accum['p_ms'] += float(p_ms.detach().cpu()) / args.grad_accum
            accum['i_anchor'] += float(i_anchor.detach().cpu()) / args.grad_accum
            if first_name is None:
                names = data['info']['name']
                first_name = names[0] if isinstance(names, list) else names

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), 3.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        print('step', step,
              'batch_size', args.batch_size,
              'grad_accum', args.grad_accum,
              'effective_batch_size', effective_batch_size,
              'clip', first_name,
              'total_loss', accum['total_loss'],
              'p_pix', accum['p_pix'],
              'p_ms', accum['p_ms'],
              'i_anchor', accum['i_anchor'],
              'lr', scheduler.get_last_lr()[0],
              'grad_norm', float(grad_norm.detach().cpu()), flush=True)

        if (step + 1) % args.save_every == 0 or step + 1 == args.steps:
            ckpt_path = args.save_dir / f'compressed_davis_stage1_encoder_step_{step + 1}.pth'
            torch.save(student.state_dict(), ckpt_path)
            print(f'saved {ckpt_path}', flush=True)


if __name__ == '__main__':
    main()
