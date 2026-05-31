import argparse
import json
import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image

from cutie.model.losses import LossComputer
from cutie.model.train_wrapper import CutieTrainWrapper


@lru_cache(maxsize=8192)
def load_rgb(path: Path, size: int) -> torch.Tensor:
    image = Image.open(path).convert('RGB').resize((size, size), Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


@lru_cache(maxsize=8192)
def load_mask(path: Path, size: int) -> np.ndarray:
    mask = Image.open(path).convert('P').resize((size, size), Image.NEAREST)
    return np.asarray(mask, dtype=np.int64)


@lru_cache(maxsize=8192)
def load_mv(path: Path, src_h: int, src_w: int, size: int) -> torch.Tensor:
    mv = np.fromfile(path, np.short).reshape(src_h, src_w, 2).astype(np.float32) / 4.0
    mv = torch.from_numpy(mv).permute(2, 0, 1).unsqueeze(0)
    mv[:, 0] *= size / src_w
    mv[:, 1] *= size / src_h
    mv = F.interpolate(mv, size=(size, size), mode='bilinear', align_corners=False)
    return mv[0].permute(1, 2, 0)


def build_clip_index(records, seq_length: int):
    by_seq = {}
    for rec in records:
        by_seq.setdefault(rec['sequence'], []).append(rec)

    clips = []
    for seq, seq_records in by_seq.items():
        seq_records = sorted(seq_records, key=lambda r: r['frame'])
        for start in range(0, len(seq_records) - seq_length + 1):
            clip = seq_records[start:start + seq_length]
            frame_ids = [int(Path(r['frame']).stem) for r in clip]
            if frame_ids == list(range(frame_ids[0], frame_ids[0] + seq_length)):
                if clip[0]['is_i_frame']:
                    clips.append(clip)
    if not clips:
        raise RuntimeError('No contiguous clip starting from an I-frame found.')
    return clips


def build_batch(clip, davis_root: Path, compressed_root: Path, size: int, max_num_obj: int):
    rgb = []
    masks = []
    mvs = []
    is_i = []
    for rec in clip:
        rgb.append(load_rgb(compressed_root / rec['decoded'], size))
        masks.append(load_mask(davis_root / 'Annotations' / 'Full-Resolution' / rec['sequence'] /
                               rec['frame'].replace('.jpg', '.png'), size))
        mvs.append(load_mv(compressed_root / rec['mv'], rec['height'], rec['width'], size))
        is_i.append(rec['is_i_frame'])

    masks = np.stack(masks, axis=0)
    labels = np.unique(masks[0])
    labels = labels[labels != 0][:max_num_obj].tolist()
    if not labels:
        raise RuntimeError(f'First frame has no objects: {clip[0]}')

    cls_gt = np.zeros((len(clip), 1, size, size), dtype=np.int64)
    first_frame_gt = np.zeros((1, max_num_obj, size, size), dtype=np.int64)
    for obj_idx, label in enumerate(labels):
        obj_mask = masks == label
        cls_gt[:, 0][obj_mask] = obj_idx + 1
        first_frame_gt[0, obj_idx] = obj_mask[0]

    selector = torch.zeros(max_num_obj, dtype=torch.float32)
    selector[:len(labels)] = 1

    return {
        'rgb': torch.stack(rgb, dim=0).unsqueeze(0),
        'first_frame_gt': torch.from_numpy(first_frame_gt).unsqueeze(0),
        'cls_gt': torch.from_numpy(cls_gt).unsqueeze(0),
        'selector': selector.unsqueeze(0),
        'mv': torch.stack(mvs, dim=0).unsqueeze(0),
        'is_i_frame': torch.tensor(is_i, dtype=torch.bool).unsqueeze(0),
        'info': {
            'num_objects': torch.tensor([len(labels)]),
            'name': clip[0]['sequence'],
            'frames': [r['frame'] for r in clip],
        },
    }


def merge_batches(batches):
    merged = {}
    for key in ['rgb', 'first_frame_gt', 'cls_gt', 'selector', 'mv', 'is_i_frame']:
        merged[key] = torch.cat([batch[key] for batch in batches], dim=0)
    merged['info'] = {
        'num_objects': torch.cat([batch['info']['num_objects'] for batch in batches], dim=0),
        'name': [batch['info']['name'] for batch in batches],
        'frames': [batch['info']['frames'] for batch in batches],
    }
    return merged


def sample_batch(clips, rng, args):
    batches = []
    used = []
    attempts = 0
    while len(batches) < args.batch_size:
        attempts += 1
        if attempts > args.batch_size * 100:
            raise RuntimeError('Too many failed clip sampling attempts.')
        clip = clips[int(rng.integers(len(clips)))]
        try:
            batch = build_batch(clip, args.davis_root, args.compressed_root, args.size,
                                args.num_objects)
        except RuntimeError as exc:
            if 'First frame has no objects' in str(exc):
                continue
            raise
        batches.append(batch)
        used.append(clip)
    return merge_batches(batches), used


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--davis-root', type=Path, default=Path('data/DAVIS/2017/trainval'))
    parser.add_argument('--compressed-root',
                        type=Path,
                        default=Path('data/DAVIS/2017/trainval/compressed/3M-GOP12'))
    parser.add_argument('--weights', type=Path, default=Path('weights/cutie-base-mega.pth'))
    parser.add_argument('--steps', type=int, default=2)
    parser.add_argument('--size', type=int, default=128)
    parser.add_argument('--seq-length', type=int, default=3)
    parser.add_argument('--num-objects', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--save-dir', type=Path, default=Path('output/creff_davis_pilot'))
    parser.add_argument('--save-every', type=int, default=100)
    parser.add_argument('--seed', type=int, default=14159265)
    parser.add_argument('--amp', action='store_true')
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    manifest = json.loads((args.compressed_root / 'manifest.json').read_text())
    clips = build_clip_index(manifest['records'], args.seq_length)
    print(f'Found {len(clips)} trainable GOP-start clips.', flush=True)

    model_cfg = OmegaConf.load('cutie/config/model/base.yaml')
    cfg = OmegaConf.create({
        'model': model_cfg,
        'use_creff': True,
        'creff_k': 7,
        'debug': True,
    })
    cfg.model.use_creff = True

    stage_cfg = OmegaConf.create({
        'name': 'compressed_smoke',
        'num_objects': args.num_objects,
        'seq_length': args.seq_length,
        'num_ref_frames': 2,
        'deep_update_prob': 0.0,
        'amp': args.amp,
        'lr_scale': 0.5,
        'freeze_decoder_for_fst': True,
        'feat_distill_weight': 1.0,
        'point_supervision': True,
        'train_num_points': 2048,
        'oversample_ratio': 3.0,
        'importance_sample_ratio': 0.75,
    })

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CutieTrainWrapper(cfg, stage_cfg).to(device).train()
    weights = torch.load(args.weights, map_location='cpu')
    model.load_weights(weights)

    loss_computer = LossComputer(cfg, stage_cfg)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=args.lr,
                                  weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == 'cuda')
    args.save_dir.mkdir(parents=True, exist_ok=True)

    for step in range(args.steps):
        data, used_clips = sample_batch(clips, rng, args)
        for key, value in list(data.items()):
            if isinstance(value, torch.Tensor):
                data[key] = value.to(device)

        optimizer.zero_grad(set_to_none=True)
        out = model(data)
        losses = loss_computer.compute({**data, **out}, out['num_filled_objects'])
        scaler.scale(losses['total_loss']).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        scaler.step(optimizer)
        scaler.update()
        first_clip = used_clips[0]
        print('step', step, 'batch_size', args.batch_size, 'clip', first_clip[0]['sequence'], [r['frame'] for r in first_clip],
              'total_loss', float(losses['total_loss'].detach().cpu()),
              'feat_distill', float(losses.get('feat_distill', torch.tensor(0.)).detach().cpu()),
              'grad_norm', float(grad_norm.detach().cpu()), flush=True)
        if (step + 1) % args.save_every == 0 or step + 1 == args.steps:
            ckpt_path = args.save_dir / f'compressed_davis_step_{step + 1}.pth'
            torch.save(model.state_dict(), ckpt_path)
            print(f'saved {ckpt_path}', flush=True)


if __name__ == '__main__':
    main()
