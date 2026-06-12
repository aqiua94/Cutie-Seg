import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm

from cutie.inference.inference_core import InferenceCore
from cutie.inference.utils.results_utils import ResultSaver
from cutie.model.cutie import CUTIE


def load_image(path: Path) -> torch.Tensor:
    image = Image.open(path).convert('RGB')
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def load_mask(path: Path) -> torch.Tensor:
    return torch.LongTensor(np.asarray(Image.open(path)))


def load_mv(path: Path, height: int, width: int) -> torch.Tensor:
    mv = np.fromfile(path, np.short).reshape(height, width, 2).astype(np.float32) / 4.0
    return torch.from_numpy(mv)


def resize_tensor_image(image: torch.Tensor, size: int) -> torch.Tensor:
    if size <= 0 or min(image.shape[-2:]) <= size:
        return image
    h, w = image.shape[-2:]
    new_h = int(h / min(h, w) * size)
    new_w = int(w / min(h, w) * size)
    return F.interpolate(image.unsqueeze(0), size=(new_h, new_w), mode='bilinear',
                         align_corners=False)[0]


def resize_mask(mask: torch.Tensor, size: int) -> torch.Tensor:
    if size <= 0 or min(mask.shape[-2:]) <= size:
        return mask
    h, w = mask.shape[-2:]
    new_h = int(h / min(h, w) * size)
    new_w = int(w / min(h, w) * size)
    return F.interpolate(mask[None, None].float(), size=(new_h, new_w), mode='nearest-exact')[0,
                                                                                              0].long()


def resize_mv(mv: torch.Tensor, size: int) -> torch.Tensor:
    if size <= 0 or min(mv.shape[:2]) <= size:
        return mv
    h, w = mv.shape[:2]
    new_h = int(h / min(h, w) * size)
    new_w = int(w / min(h, w) * size)
    mv = mv.permute(2, 0, 1).unsqueeze(0)
    mv[:, 0] *= new_w / w
    mv[:, 1] *= new_h / h
    mv = F.interpolate(mv, size=(new_h, new_w), mode='bilinear', align_corners=False)
    return mv[0].permute(1, 2, 0)


def build_records_by_sequence(compressed_root: Path):
    manifest = json.loads((compressed_root / 'manifest.json').read_text())
    by_seq = {}
    for rec in manifest['records']:
        by_seq.setdefault(rec['sequence'], {})[rec['frame']] = rec
    return by_seq


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--davis-root', type=Path, default=Path('data/DAVIS/2017/trainval'))
    parser.add_argument('--compressed-root',
                        type=Path,
                        default=Path('data/DAVIS/2017/trainval/compressed/3M-GOP12'))
    parser.add_argument('--subset', type=Path, default=Path('data/DAVIS/2017/trainval/ImageSets/2017/val.txt'))
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=Path('output/creff_davis_eval'))
    parser.add_argument('--size', type=int, default=480)
    parser.add_argument('--max-videos', type=int, default=-1)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--disable-creff', action='store_true')
    parser.add_argument('--compressed-p-frames', action='store_true',
                        help='Encode non-I GOP frames at LR without requiring CReFF injection.')
    parser.add_argument('--gop-length', type=int, default=12)
    parser.add_argument('--lr-scale', type=float, default=0.5)
    parser.add_argument('--creff-k', type=int, default=7)
    parser.add_argument('--lr-downstream', action='store_true')
    args = parser.parse_args()

    cfg = OmegaConf.load('cutie/config/eval_config.yaml')
    cfg.model = OmegaConf.load('cutie/config/model/base.yaml')
    cfg.model.use_creff = not args.disable_creff
    cfg.model.creff_k = args.creff_k
    cfg.use_creff = not args.disable_creff
    cfg.creff_k = args.creff_k
    cfg.compressed_p_frames = args.compressed_p_frames or cfg.use_creff
    cfg.gop_length = args.gop_length
    cfg.lr_scale = args.lr_scale
    cfg.lr_downstream = args.lr_downstream
    cfg.hr_pix_memory_keep = 2
    cfg.mem_every = 5
    cfg.stagger_updates = 5
    cfg.chunk_size = -1
    cfg.max_internal_size = -1
    cfg.flip_aug = False
    cfg.save_aux = False
    cfg.save_scores = False
    cfg.visualize = False
    cfg.long_term.use_long_term = False
    cfg.max_mem_frames = 5

    cutie = CUTIE(cfg).cuda().eval()
    cutie.load_weights(torch.load(args.weights, map_location='cpu'))
    records_by_sequence = build_records_by_sequence(args.compressed_root)

    videos = [line.strip() for line in args.subset.read_text().splitlines() if line.strip()]
    if args.max_videos > 0:
        videos = videos[:args.max_videos]

    mask_output_root = args.output_dir / 'Annotations'
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for video in tqdm(videos):
        seq_records = records_by_sequence.get(video, {})
        if not seq_records:
            print(f'skip {video}: no compressed records', flush=True)
            continue

        mask_dir = args.davis_root / 'Annotations' / 'Full-Resolution' / video
        image_dir = args.davis_root / 'JPEGImages' / 'Full-Resolution' / video
        frames = sorted(p.name for p in image_dir.glob('*.jpg'))
        if not frames:
            frames = sorted(seq_records)
        first_mask = Image.open(mask_dir / frames[0].replace('.jpg', '.png'))
        palette = first_mask.getpalette()
        processor = InferenceCore(cutie, cfg=cfg)
        saver = ResultSaver(str(mask_output_root),
                            video,
                            dataset='d17-val',
                            object_manager=processor.object_manager,
                            use_long_id=False,
                            palette=palette)

        try:
            for ti, frame in enumerate(frames):
                rec = seq_records.get(frame)
                if rec is not None:
                    image_path = args.compressed_root / rec['decoded']
                    image = load_image(image_path)
                    mv = load_mv(args.compressed_root / rec['mv'], rec['height'], rec['width'])
                else:
                    image_path = image_dir / frame
                    image = load_image(image_path)
                    mv = None
                mask_path = mask_dir / frame.replace('.jpg', '.png')
                mask = load_mask(mask_path) if ti == 0 else None
                valid_labels = None
                if mask is not None:
                    valid_labels = torch.unique(mask)
                    valid_labels = valid_labels[valid_labels != 0].tolist()

                orig_shape = image.shape[-2:]
                image = resize_tensor_image(image, args.size).cuda()
                if mv is not None:
                    mv = resize_mv(mv, args.size).cuda()
                if mask is not None:
                    mask = resize_mask(mask, args.size).cuda()

                with torch.cuda.amp.autocast(enabled=args.amp):
                    prob = processor.step(image,
                                          mask,
                                          valid_labels,
                                          end=(ti == len(frames) - 1),
                                          mv=mv)
                saver.process(prob,
                              frame,
                              resize_needed=(tuple(prob.shape[-2:]) != tuple(orig_shape)),
                              shape=orig_shape,
                              last_frame=(ti == len(frames) - 1),
                              path_to_image=str(image_path))
        finally:
            saver.end()

    print(f'saved masks to {mask_output_root}', flush=True)


if __name__ == '__main__':
    main()
