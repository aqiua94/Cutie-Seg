import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from omegaconf import OmegaConf

from cutie.inference.inference_core import InferenceCore
from cutie.model.cutie import CUTIE
from scripts.eval_compressed_davis import (build_records_by_sequence, load_image, load_mask, load_mv,
                                           resize_mask, resize_mv, resize_tensor_image)


MODES = {
    "a0_hr480": dict(size=480, disable_creff=True, compressed_p_frames=False, lr_scale=0.5),
    "all240": dict(size=240, disable_creff=True, compressed_p_frames=False, lr_scale=1.0),
    "a1_hr480_lr240": dict(size=480, disable_creff=True, compressed_p_frames=True, lr_scale=0.5),
    "b3b_hr480_lr240_creff": dict(size=480, disable_creff=False, compressed_p_frames=True, lr_scale=0.5),
}


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    if not values:
        return 0.0
    index = (len(values) - 1) * q
    lo = int(index)
    hi = min(lo + 1, len(values) - 1)
    frac = index - lo
    return values[lo] * (1.0 - frac) + values[hi] * frac


def summarize(values: list[float]) -> dict[str, float | int]:
    total_ms = sum(values)
    return {
        "frames": len(values),
        "total_ms": total_ms,
        "fps": len(values) * 1000.0 / total_ms if total_ms else 0.0,
        "mean_ms": statistics.fmean(values) if values else 0.0,
        "p50_ms": percentile(values, 0.50),
        "p95_ms": percentile(values, 0.95),
    }


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--davis-root", type=Path, default=Path("data/DAVIS/2017/trainval"))
    parser.add_argument("--compressed-root", type=Path,
                        default=Path("data/DAVIS/2017/trainval/compressed/3M-GOP12"))
    parser.add_argument("--subset", type=Path,
                        default=Path("data/DAVIS/2017/trainval/ImageSets/2017/val.txt"))
    parser.add_argument("--gop-length", type=int, default=12)
    parser.add_argument("--creff-k", type=int, default=7)
    parser.add_argument("--max-videos", type=int, default=-1)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    mode = MODES[args.mode]
    cfg = OmegaConf.load("cutie/config/eval_config.yaml")
    cfg.model = OmegaConf.load("cutie/config/model/base.yaml")
    cfg.model.use_creff = not mode["disable_creff"]
    cfg.model.creff_k = args.creff_k
    cfg.use_creff = not mode["disable_creff"]
    cfg.creff_k = args.creff_k
    cfg.compressed_p_frames = mode["compressed_p_frames"] or cfg.use_creff
    cfg.gop_length = args.gop_length
    cfg.lr_scale = mode["lr_scale"]
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

    network = CUTIE(cfg).cuda().eval()
    network.load_weights(torch.load(args.weights, map_location="cpu"))
    records_by_sequence = build_records_by_sequence(args.compressed_root)
    videos = [line.strip() for line in args.subset.read_text().splitlines() if line.strip()]
    if args.max_videos > 0:
        videos = videos[:args.max_videos]

    all_ms: list[float] = []
    i_ms: list[float] = []
    p_ms: list[float] = []
    sequence_rows = []
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    for video in videos:
        seq_records = records_by_sequence[video]
        mask_dir = args.davis_root / "Annotations" / "Full-Resolution" / video
        image_dir = args.davis_root / "JPEGImages" / "Full-Resolution" / video
        frames = sorted(path.name for path in image_dir.glob("*.jpg"))
        processor = InferenceCore(network, cfg=cfg)
        sequence_ms = []

        for ti, frame in enumerate(frames):
            rec = seq_records.get(frame)
            if rec is not None:
                image = load_image(args.compressed_root / rec["decoded"])
                mv = load_mv(args.compressed_root / rec["mv"], rec["height"], rec["width"])
            else:
                image = load_image(image_dir / frame)
                mv = None
            mask = load_mask(mask_dir / frame.replace(".jpg", ".png")) if ti == 0 else None
            valid_labels = None
            if mask is not None:
                valid_labels = torch.unique(mask)
                valid_labels = valid_labels[valid_labels != 0].tolist()

            image = resize_tensor_image(image, mode["size"]).cuda()
            if mv is not None:
                mv = resize_mv(mv, mode["size"]).cuda()
            if mask is not None:
                mask = resize_mask(mask, mode["size"]).cuda()

            start = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start.record()
            with torch.amp.autocast("cuda", enabled=True):
                processor.step(image, mask, valid_labels, end=(ti == len(frames) - 1), mv=mv)
            end_event.record()
            end_event.synchronize()
            elapsed_ms = start.elapsed_time(end_event)
            all_ms.append(elapsed_ms)
            sequence_ms.append(elapsed_ms)
            is_i_frame = mask is not None or ti % args.gop_length == 0
            (i_ms if is_i_frame else p_ms).append(elapsed_ms)

        sequence_rows.append({"sequence": video, **summarize(sequence_ms)})

    result = {
        "mode": args.mode,
        "weights": str(args.weights),
        "geometry": mode,
        "gop_length": args.gop_length,
        "amp": True,
        "timing_scope": "GPU time for InferenceCore.step only; excludes disk IO, input resize, and mask saving",
        "overall": summarize(all_ms),
        "i_frames": summarize(i_ms),
        "p_frames": summarize(p_ms),
        "peak_memory_mib": torch.cuda.max_memory_allocated() / (1024**2),
        "sequences": sequence_rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    csv_path = args.output_json.with_suffix(".sequences.csv")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sequence_rows[0].keys())
        writer.writeheader()
        writer.writerows(sequence_rows)
    print(json.dumps({key: result[key] for key in ("mode", "geometry", "overall", "i_frames", "p_frames", "peak_memory_mib")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
