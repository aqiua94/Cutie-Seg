# Training Cutie

## Setting Up Data

We put datasets out-of-source, as in XMem. You do not need BL30K. The directory structure should look like this:

```bash
├── Cutie
├── DAVIS
│   └── 2017
│       ├── test-dev
│       │   ├── Annotations
│       │   └── ...
│       └── trainval
│           ├── Annotations
│           └── ...
├── BURST
│   ├── frames
│   ├── val
│   │   ├── all_classes.json
│   │   └── first_frame_annotations.json
│   ├── train
│   │   └── train.json
│   └── train-vos
│       ├── JEPGImages
│       └── Annotations
├── static
│   ├── BIG_small
│   └── ...
└── YouTube
│   ├── all_frames
│   │   └── valid_all_frames
│   ├── train
│   └── valid
├── OVIS-VOS-train
│   ├── JPEGImages
│   └── Annotations
└── MOSE
    ├── JPEGImages
    └── Annotations
```

DEVA has a script for downloading some of these datasets: <https://github.com/hkchengrex/Tracking-Anything-with-DEVA/blob/main/docs/TRAINING.md>.

To generate `train-vos` for BURST, use the script `scripts/convert_burst_to_vos_train.py` which extracts masks from the JSON file into the DAVIS/YouTubeVOS format for training:
```bash
python scripts/convert_burst_to_vos_train.py --json_path ../BURST/train/train.json --frames_path ../BURST/frames/train --output_path ../BURST/train-vos
```

To generate OVIS-VOS-train, use something like https://github.com/youtubevos/vis2vos or download our preprocessed version from https://drive.google.com/uc?id=1AZPyyqVqOl6j8THgZ1UdNJY9R1VGEFrX.

Links to the datasets:
- DAVIS: https://davischallenge.org/
- YouTubeVOS: https://youtube-vos.org/
- BURST: https://github.com/Ali2500/BURST-benchmark
- MOSE: https://henghuiding.github.io/MOSE/
- LVOS: https://lingyihongfd.github.io/lvos.github.io/
- OVIS: https://songbai.site/ovis/

## Training Command

We trained with four A100 GPUs, which took around 30 hours.

```
OMP_NUM_THREADS=4 torchrun --master_port 25357 --nproc_per_node=4 cutie/train.py exp_id=[some unique id] model=[small/base] data=[base/with-mose/mega]
```

- Change `nproc_per_node` to change the number of GPUs.
- Prepend `CUDA_VISIBLE_DEVICES=...` if you want to use specific GPUs.
- Change `master_port` if you encounter port collision.
- `exp_id` is a unique experiment identifier that does not affect how the training is done.
- Models and visualizations will be saved in `./output/`.
- For pre-training only, specify `main_training.enabled=False`.
- For main training only, specify `pre_training.enabled=False`.
- To load a pre-trained model, e.g., to continue main training from the final model from pre-training, specify `weights=[path to the model]`.

## Final All-Stage Size-480 Experiment

This run is the strict all-stage-size480 comparison requested for alignment with the Cutie training resolution. The compressed source RGB and MV remain full-resolution on disk; the dataloader downsamples the short side and crops RGB, masks, and correctly rescaled MV to 480x480.

Important interpretation: Stage1 at size480 applies task-only finetuning to 480 inputs. It is an all-stage-size480/Cutie-style task-finetune ablation, not pure LR=240 encoder adaptation. Stage2 and final evaluation use I/HR frames at 480 and P/LR frames at 240 through lr_scale=0.5.

Preflight and anchors:

- Native compressed-frame minimum short side: 720, so size480 never upsamples the source.
- A0 official HR upper: J&F 0.8630, from output/baseline_official_no_creff_gop1_fullval.
- A1 official true LR lower: J&F 0.6486, from output/baseline_official_no_creff_gop12_lr_pframes_fullval. Its precise deployment geometry is HR480/LR240: `--size 480` establishes the I-frame/HR canvas, while `--compressed-p-frames --lr-scale 0.5` downsamples non-I frames to 240 inside `InferenceCore` before encoding. There is no separate official A1@240 true-lower run with J&F 0.6486.
- The independent all-240 no-CReFF baseline encodes every frame at short-side 240 and reaches J&F 0.8266; it is not the true-lower deployment path.
- 70% gap-recovery success line: J&F 0.7987.
- Stage1 s480 smoke passed at b2a16/eff_bs32: layer3 grad present, res2 and mask decoder frozen, feat_distill=0, step0 total loss 1.6297. The earlier proposed 0.4-0.6 loss sanity range does not match this repository loss composition.

Single-GPU execution order:

1. B1: official Cutie to task-only Stage1 at size480 for 1500 steps; validate true-LR path at steps 500/1000/1500.
2. Gate: B1 step1500 must exceed A1 before B3 starts.
3. B3: B1 step1500 encoder to CReFF-only Stage2 at size480 for 3000 steps; validate every 500 steps.
4. B2: official Cutie to CReFF-only Stage2 at size480 for 3000 steps; validate every 500 steps as the strict B3 control.

All training uses seed 14159265, seq8, GOP12, k7, no AMP, b2a16/eff_bs32, workers8/prefetch4. Full validation always covers all 30 DAVIS-2017 val sequences at eval size480. The complete resumable pipeline is scripts/run_final_s480_from_scratch.sh and writes to output/final_B1_stage1_s480, output/final_B3_stage1plus2_s480, and output/final_B2_creff_only_s480.

### Stage Checkpoint Selection Rule

Never initialize the next training stage from the final checkpoint by default. Select the upstream-stage checkpoint using its validation curve and record the selection metric. A final checkpoint is valid only when it is also the validation peak or when a deliberate last-checkpoint ablation is being run. Stage1@240 was monotonic, so step1500 was both final and best. Stage1@480 peaked at step500 and regressed afterward; using step1500 to initialize B3 introduced a checkpoint-selection confound that B3b explicitly tests.

### Geometry Comparison And Noise-Control Order

B3b initialized Stage2@480/240 from the Stage1@480 validation-peak step500 checkpoint and reached J&F 0.7545. This is +0.38 points over B3 initialized from the regressed Stage1 step1500 checkpoint, so peak selection helps in the expected direction but remains inside the preregistered 0.5-point noise branch. Treat checkpoint selection as a small confound, not as the explanation for the earlier B3-versus-Q1 ordering.

Stage1 benefit is established across both tested geometries: B3b exceeds B2 by 1.55 points at train geometry 480/240, and Q1 exceeds Q2 by 2.05 points at train geometry 240/120. Geometry ranking is not established because B3b 0.7545 and Q1 0.7563 differ by only 0.18 points.

B3b second-seed noise control is complete. The two B3b final step3000 results are 0.7545 and 0.7563, a difference of 0.18 points, with a two-seed mean of 0.7554. Q1 at 0.7563 lies inside this observed final-result range, so Q1 versus B3b cannot rank the tested geometries. Treat endpoint reproducibility and trajectory variability separately: the matched final-step difference is 0.18 points, while intermediate checkpoints differ by up to 0.98 points. The latter shows seed-sensitive convergence speed, not a measured +/-0.5-to-1.0-point final-result noise band. Both B3b runs load the same frozen Stage1 checkpoint, so this control measures Stage2 CReFF initialization and dataloader-order variability only.

A1@480 is already complete at J&F 0.6486 and should not be rerun unless the evaluation protocol changes. B3c/H1 is optional and can only strengthen the geometry-robustness coverage; it is not expected to rank geometries. Prioritize an explicit A0/A1/B3b FPS benchmark. A B3b step3000-to-5000 extension is also optional, but model checkpoints do not contain optimizer state, so the extension resets AdamW and must be labeled as an optimizer-reset continuation rather than a strictly continuous convergence test.


### Full-Val FPS Benchmark Result

The four-way GPU-compute benchmark is complete on all 30 DAVIS-2017 val sequences / 1999 frames. CUDA-event timing covers `InferenceCore.step()` only with AMP and excludes disk IO, input resize, and mask saving.

| Configuration | J&F | FPS | Peak memory MiB |
|---|---:|---:|---:|
| A0 HR480 every frame | 0.8630 | 62.85 | 1019.95 |
| all-240 every frame | 0.8266 | 69.67 | 399.54 |
| A1 HR480/LR240, no CReFF | 0.6486 | 64.40 | 1055.00 |
| B3b HR480/LR240 + CReFF | 0.7554 | 60.38 | 1055.04 |

All-240 Pareto-dominates the current B3b implementation: 1.154x faster, +7.12 J&F points, and 0.379x peak memory. B3b is also slower than A0 HR480. Stop treating FPS as an unverified method advantage. Before spending GPU on B3c or longer accuracy training, profile and redesign the compressed path so it can produce a measurable runtime benefit, or revise the method framing away from speed efficiency.

### CPU Threading Rule For Multi-Worker Loading

When num_workers is greater than zero, launch training with OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1, and NUMEXPR_NUM_THREADS=1. Otherwise every DataLoader worker may inherit the host-wide thread count and oversubscribe CPU resize/interpolation operations. An observed B3b launch inherited OMP/MKL=25 with 8 workers and drove load average above 100 on a 25-vCPU host. The run was stopped before its first checkpoint and restarted with per-process thread counts fixed to one.

