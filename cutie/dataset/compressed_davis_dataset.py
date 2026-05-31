import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


class CompressedDAVISDataset(Dataset):
    """DAVIS clips with decoded HEVC frames and MV-to-I fields."""

    def __init__(
        self,
        davis_root: Path,
        compressed_root: Path,
        seq_length: int = 3,
        size: int = 480,
        max_num_obj: int = 3,
        seed: int = 14159265,
        resize_mode: str = 'crop',
    ):
        self.davis_root = Path(davis_root)
        self.compressed_root = Path(compressed_root)
        self.seq_length = seq_length
        self.size = size
        self.max_num_obj = max_num_obj
        self.resize_mode = resize_mode
        self.rng = np.random.default_rng(seed)
        if self.resize_mode not in {'crop', 'square'}:
            raise ValueError(f'Unknown resize_mode: {self.resize_mode}')

        manifest = json.loads((self.compressed_root / 'manifest.json').read_text())
        self.clips = self._build_clip_index(manifest['records'])
        if not self.clips:
            raise RuntimeError('No contiguous clip starting from an I-frame found.')

    def _build_clip_index(self, records: List[Dict]) -> List[List[Dict]]:
        by_seq = {}
        for rec in records:
            by_seq.setdefault(rec['sequence'], []).append(rec)

        clips = []
        for _, seq_records in by_seq.items():
            seq_records = sorted(seq_records, key=lambda r: r['frame'])
            for start in range(0, len(seq_records) - self.seq_length + 1):
                clip = seq_records[start:start + self.seq_length]
                frame_ids = [int(Path(r['frame']).stem) for r in clip]
                expected = list(range(frame_ids[0], frame_ids[0] + self.seq_length))
                if frame_ids == expected and clip[0]['is_i_frame']:
                    clips.append(clip)
        return clips

    def _resize_shape(self, height: int, width: int) -> tuple[int, int]:
        if self.resize_mode == 'square':
            return self.size, self.size

        scale = self.size / min(height, width)
        return int(round(height * scale)), int(round(width * scale))

    def _sample_crop(self, height: int, width: int) -> tuple[int, int]:
        if self.resize_mode == 'square':
            return 0, 0

        top_max = max(height - self.size, 0)
        left_max = max(width - self.size, 0)
        top = int(np.random.randint(0, top_max + 1)) if top_max > 0 else 0
        left = int(np.random.randint(0, left_max + 1)) if left_max > 0 else 0
        return top, left

    def _crop_array(self, array: np.ndarray, top: int, left: int) -> np.ndarray:
        if self.resize_mode == 'square':
            return array
        return array[top:top + self.size, left:left + self.size]

    def _load_rgb(self, path: Path, resize_shape: tuple[int, int],
                  crop: tuple[int, int]) -> torch.Tensor:
        resize_h, resize_w = resize_shape
        top, left = crop
        image = Image.open(path).convert('RGB').resize((resize_w, resize_h), Image.BILINEAR)
        if self.resize_mode == 'crop':
            image = image.crop((left, top, left + self.size, top + self.size))
        array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1)

    def _load_mask(self, path: Path, resize_shape: tuple[int, int],
                   crop: tuple[int, int]) -> np.ndarray:
        resize_h, resize_w = resize_shape
        top, left = crop
        mask = Image.open(path).convert('P').resize((resize_w, resize_h), Image.NEAREST)
        mask_array = np.asarray(mask, dtype=np.int64)
        return self._crop_array(mask_array, top, left)

    def _load_mv(self, path: Path, src_h: int, src_w: int, resize_shape: tuple[int, int],
                 crop: tuple[int, int]) -> torch.Tensor:
        resize_h, resize_w = resize_shape
        top, left = crop
        mv = np.fromfile(path, np.short).reshape(src_h, src_w, 2).astype(np.float32) / 4.0
        mv = torch.from_numpy(mv).permute(2, 0, 1).unsqueeze(0)
        mv[:, 0] *= resize_w / src_w
        mv[:, 1] *= resize_h / src_h
        mv = F.interpolate(mv, size=(resize_h, resize_w), mode='bilinear', align_corners=False)
        mv = mv[0].permute(1, 2, 0)
        if self.resize_mode == 'crop':
            mv = mv[top:top + self.size, left:left + self.size]
        return mv

    def _read_clip(self, clip: List[Dict]) -> Dict:
        rgb = []
        masks = []
        mvs = []
        is_i = []
        resize_shape = self._resize_shape(clip[0]['height'], clip[0]['width'])
        crop = self._sample_crop(*resize_shape)
        for rec in clip:
            frame_png = rec['frame'].replace('.jpg', '.png')
            rgb.append(self._load_rgb(self.compressed_root / rec['decoded'], resize_shape, crop))
            masks.append(
                self._load_mask(self.davis_root / 'Annotations' / 'Full-Resolution' /
                                rec['sequence'] / frame_png, resize_shape, crop))
            mvs.append(
                self._load_mv(self.compressed_root / rec['mv'], rec['height'], rec['width'],
                              resize_shape, crop))
            is_i.append(rec['is_i_frame'])

        masks = np.stack(masks, axis=0)
        labels = np.unique(masks[0])
        labels = labels[labels != 0][:self.max_num_obj].tolist()
        if not labels:
            raise RuntimeError('First frame has no objects.')

        cls_gt = np.zeros((self.seq_length, 1, self.size, self.size), dtype=np.int64)
        first_frame_gt = np.zeros((1, self.max_num_obj, self.size, self.size), dtype=np.int64)
        for obj_idx, label in enumerate(labels):
            obj_mask = masks == label
            cls_gt[:, 0][obj_mask] = obj_idx + 1
            first_frame_gt[0, obj_idx] = obj_mask[0]

        selector = torch.zeros(self.max_num_obj, dtype=torch.float32)
        selector[:len(labels)] = 1

        return {
            'rgb': torch.stack(rgb, dim=0),
            'first_frame_gt': torch.from_numpy(first_frame_gt),
            'cls_gt': torch.from_numpy(cls_gt),
            'selector': selector,
            'mv': torch.stack(mvs, dim=0),
            'is_i_frame': torch.tensor(is_i, dtype=torch.bool),
            'info': {
                'num_objects': torch.tensor(len(labels), dtype=torch.long),
                'name': clip[0]['sequence'],
                'frames': [r['frame'] for r in clip],
            },
        }

    def __getitem__(self, idx: int) -> Dict:
        for _ in range(100):
            clip = self.clips[idx % len(self.clips)]
            try:
                return self._read_clip(clip)
            except RuntimeError as exc:
                if 'First frame has no objects' not in str(exc):
                    raise
                idx = int(self.rng.integers(len(self.clips)))
        raise RuntimeError('Too many failed clip sampling attempts.')

    def __len__(self) -> int:
        return len(self.clips)
