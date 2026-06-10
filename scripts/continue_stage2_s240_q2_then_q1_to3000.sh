#!/usr/bin/env bash
set -euo pipefail

PY=/root/miniconda3/bin/python
COMMON_TRAIN=(--target-step 3000 --size 240 --resize-mode crop --seq-length 8 --batch-size 4 --grad-accum 8 --num-workers 8 --prefetch-factor 4 --lr 1e-4 --save-every 500 --creff-k 7 --trainable-mode creff_only --feat-distill-type mse_creff_to_hr --feat-distill-weight 1.0)

run_eval_curve() {
  local save_dir=$1
  local eval_prefix=$2
  local method=$3
  local notes=$4
  for step in 2000 2500 3000; do
    $PY scripts/watch_checkpoint_eval_jf.py --checkpoint ${save_dir}/compressed_davis_dataloader_step_${step}.pth --step ${step} --method "${method}" --output-dir ${eval_prefix}_step${step} --size 480 --gop-length 12 --creff-k 7 --workers 12 --amp --train-batch-size 4 --train-grad-accum 8 --train-seq-length 8 --train-size-desc 240x240_crop --trainable-desc "CReFF only" --frozen-desc "frozen encoder/decoder; P frames downsampled to 120 during training" --train-lr-desc "lr_creff=1e-4; AdamW reset at step1500 continuation" --notes "${notes}"
  done
}

Q2_DIR=output/stage2_s240_official_seed14159265
$PY scripts/resume_compressed_davis_dataloader_until.py --weights weights/cutie-base-mega.pth --save-dir ${Q2_DIR} "${COMMON_TRAIN[@]}"
run_eval_curve ${Q2_DIR} output/eval_stage2_s240_official "Q2 CReFF-only official encoder train-size240" "same-size control; continued from step1500 with shared AdamW reset; eval final GOP12 path at size480"

Q1_DIR=output/stage2_after_stage1_s240_seed14159265
$PY scripts/resume_compressed_davis_dataloader_until.py --weights output/stage1_plan34_lr_adapt_s240_l8_b4a8_seed14159265/stage1_plan34_step_1500.pth --save-dir ${Q1_DIR} "${COMMON_TRAIN[@]}"
run_eval_curve ${Q1_DIR} output/eval_stage2_after_stage1_s240 "Q1 Stage2-on-Stage1 train-size240" "Stage1 plan3.4 encoder plus CReFF-only Stage2; continued from step1500 with shared AdamW reset; eval final GOP12 path at size480"

$PY scripts/plot_stage2_q1_q2_curve.py
