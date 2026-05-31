# Cutie + AR-Seg 训练策略

> 配套文档：[Cutie_AR-Seg融合策略.md](./Cutie_AR-Seg融合策略.md)（架构 / 代码改造）
> 本文档专注训练流程；训练进展会持续更新在第十节。

---

## 一、总原则

**不从头训练**。官方 `cutie-base-mega.pth` 已经吃了 static images + BL30K + DAVIS + YouTube-VOS + MOSE 的完整四阶段训练，重复跑一遍要 2-3 周 × 8 卡，无意义。

我们要做的只有两件事：
1. 让 `pixel_encoder` 适应**降分辨率**输入
2. 训练新增的 `CReFFBlock`，让它学会"用 HR 参考帧补 LR 当前帧的细节"

---

## 二、三阶段总览

| 阶段 | 起点权重 | 训练对象 | 冻结对象 | 目的 | 预计时长（A100 单卡） |
|---|---|---|---|---|---|
| **Stage 0** | — | 不训，直接下载 | 全部 | 拿 baseline 权重 | 0（下载 5 分钟） |
| **Stage 1** | `cutie-base-mega.pth` | `pixel_encoder` | mask_encoder + key_proj + pixel_fuser + object_transformer + object_summarizer + mask_decoder | encoder 适应 LR 输入分布 | ~30 epoch / 1-2 天 |
| **Stage 2** | Stage 1 输出 | `pixel_encoder` + `CReFFBlock` | 同上 + key_proj（一定要冻，保证 memory match 一致性） | 学 CReFF 的 MV 补偿 + FST 蒸馏 | ~50 epoch / 2-3 天 |

> **省钱替代方案**：算力紧时可以跳过 Stage 1，直接 Stage 2，但学习率要降到 5e-6，多 20 epoch 让 encoder 慢慢适应；论文里建议两阶段以体现 ablation 清晰度。

---

## 三、Stage 1：LR 适配训练

### 3.1 数据
- **训练集**：DAVIS-2017 train (60 seq) + YouTube-VOS 2019 train (3471 seq) + MOSEv1 train (1507 seq)
- **预处理**：用 AR-Seg 的 x265 工具链把原视频重编成 HEVC@3Mbps + GOP=12（与 Stage 2 保持一致，便于 cache 复用）
- **采样**：每个 clip 长度 `seq_length=8`（Cutie 主训默认），起始帧随机
- **分辨率**：原始 480p（DAVIS）/ 720p（YT-VOS）输入后下采到 0.5×，再 pad 到 16 的倍数

### 3.2 网络配置
```yaml
# config/training/stage1.yaml
model:
  use_creff: false          # Stage 1 不开 CReFF
stage_cfg:
  num_objects: 3
  seq_length: 8
  num_ref_frames: 3
  lr_scale: 0.5
  gop_length: 12            # 数据预处理一致即可
  freeze_decoder: true
  feat_loss_weight: 0.0     # Stage 1 无蒸馏
```

冻结代码（在 `train_wrapper.py` 里）：
```python
for m in [self.mask_encoder, self.key_proj, self.pixel_fuser,
          self.object_transformer, self.object_summarizer, self.mask_decoder]:
    for p in m.parameters(): p.requires_grad = False
```

### 3.3 优化器
| 项 | 值 | 备注 |
|---|---|---|
| Optimizer | AdamW | β=(0.9, 0.999), weight_decay=0.05 |
| LR (pixel_encoder) | **1e-5** | 比 Cutie 默认 1e-4 小 10×，避免破坏预训练 |
| LR schedule | linear warmup 500 steps → cosine | T_max = total_iters |
| Batch size | 8 (per GPU) × num_objects=3 | A100-80G 够 |
| Mixed precision | AMP fp16 | 沿用 Cutie 默认 |
| Gradient clip | 0.5 | 沿用 Cutie 默认 |
| Epochs | 30 | 早停看 DAVIS val J&F |

### 3.4 Loss
完全用 Cutie 原版：
```
total_loss = bootstrapped_CE + Dice + 0.01 * sensory_aux + 0.01 * query_aux
```
**不加** MSE 蒸馏（这阶段没有 teacher / student 概念）。

### 3.5 验证 & 成功标准
- 每 epoch 在 DAVIS-2017 val 上测一次 J&F（用单分辨率 LR 推理，不走 CReFF）
- **目标**：J&F ≥ 原 HR Cutie 的 95%（即 baseline 86 → LR 模型 81+ 即可。下降是预期的，CReFF 会在 Stage 2 补回来）
- 若 J&F 在前 10 epoch 就跌到 75 以下，说明 LR 0.5× 太激进，回退到 0.7×

---

## 四、Stage 2：CReFF + FST 训练

### 4.1 数据
同 Stage 1，但 **dataloader 必须返回 MV 和 is_i_frame**：
```python
batch = {
    'rgb':            [B, T, 3, H, W],          # 全 HR，模型内部按 is_i 决定要不要下采
    'first_frame_gt': [B, num_objects, H, W],
    'mv':             [B, T, H_mv, W_mv, 2],    # I 帧位置全 0 占位
    'is_i_frame':     [B, T] bool,              # 第一帧强制 True
    'info':           {...},
}
```

### 4.2 网络配置
```yaml
# config/training/stage2.yaml
model:
  use_creff: true
  creff_k: 7                # 7×7 局部 attention，AR-Seg 默认
stage_cfg:
  num_objects: 3
  seq_length: 8
  num_ref_frames: 3
  lr_scale: 0.5
  gop_length: 8             # 训练时 GOP 短一点，每个 clip 至少 1 个 I 帧
  freeze_decoder: true
  feat_loss_weight: 1.0     # FST 蒸馏权重
```

冻结代码：
```python
for m in [self.mask_encoder, self.key_proj, self.pixel_fuser,
          self.object_transformer, self.object_summarizer, self.mask_decoder]:
    for p in m.parameters(): p.requires_grad = False
# pixel_encoder 和 self.creff 可训
```

### 4.3 优化器
| 项 | 值 | 备注 |
|---|---|---|
| Optimizer | AdamW | 同 Stage 1 |
| LR (pixel_encoder) | **5e-6** | 再降一档，因为有 teacher 约束 |
| LR (CReFFBlock) | **1e-4** | CReFF 是新模块，需要正常 LR |
| 参数组 | 两组 | `optim.AdamW([{params: enc, lr: 5e-6}, {params: creff, lr: 1e-4}])` |
| Batch size | 6 (per GPU) | 比 Stage 1 略小，因为多了 teacher 一次 forward |
| Epochs | 50 | 看蒸馏 loss 何时 plateau |

### 4.4 Loss
```python
total_loss = (
    bootstrapped_CE                       # 原 Cutie
    + Dice                                # 原 Cutie
    + 0.01 * sensory_aux                  # 原 Cutie
    + 0.01 * query_aux                    # 原 Cutie
    + 1.0  * mse_feat_distill             # 新增：MSE(pix_feat_HR_teacher, pix_feat_LR_after_creff)
)
```

蒸馏 loss 只对 **P 帧** 计算（I 帧的 teacher 就是它自己，没意义）：
```python
mse_pairs = [(teacher[t], student[t]) for t in range(T) if not is_i[t]]
feat_loss = sum(F.mse_loss(s, t.detach()) for t, s in mse_pairs) / len(mse_pairs)
```

### 4.5 验证 & 成功标准
- DAVIS-2017 val J&F：**目标 ≥ HR Cutie 的 98%**（即 86 → 84+ 可接受，85+ 为好）
- MOSEv1 val J&F：**目标 ≥ HR Cutie 的 95%**（复杂场景容忍度放宽）
- FPS：**目标 ≥ HR Cutie 的 2×**（CReFF 的卖点）
- 若 J&F 跌幅 > 3 个点，检查：
  - MV 缩放是否分别缩 H/W（AR-Seg 原版只缩 H，对 16:9 数据集会算错）
  - CReFFBlock 的 LR 路径残差连接是否生效（`return lr_feat + out`）
  - decoder 是否真的冻死了（`grep requires_grad=False` 全套）

---

## 五、硬件与资源估算

| 配置 | 显存占用 | 单 epoch 时间（DAVIS+YT+MOSE 全集） | 备注 |
|---|---|---|---|
| **A100-80G ×1** | ~50G | ~1.5 h | 推荐起步 |
| **A100-80G ×4 (DDP)** | ~50G/卡 | ~25 min | 主训用 |
| **RTX 4090 ×1** | OOM 风险 | ~3 h | 把 seq_length 降到 5，batch 降到 4 才行 |
| **3090 ×1** | OOM | — | 不建议，要么改 seq_length=3 |

**总训练时间预算**（A100 ×4）：
- Stage 1: 30 epoch × 25 min ≈ 12.5 h
- Stage 2: 50 epoch × 30 min ≈ 25 h（teacher forward 多一份）
- 合计：约 **1.5 天**

---

## 六、Debug / Sanity Check 清单

在开始正式训练前必做：

### 6.1 数据流
- [ ] 随机抽 10 个 batch，可视化 `is_i_frame` 序列，确认第 0 帧必为 True，之后每 `gop_length` 帧出现一次 True
- [ ] 可视化 MV：把 `mv[B, t, :, :, 0]` 当 R 通道、`mv[B, t, :, :, 1]` 当 G 通道画出来，应能看到主要物体的运动方向
- [ ] 检查 MV 数值范围：解码后除以 4，单位应是像素，绝对值通常 < 50

### 6.2 模型前向
- [ ] 加载官方 cutie-base-mega.pth + `use_creff=True`，跑一遍 inference 全 I 帧（gop_length=1），J&F 应该 = 原 Cutie 论文报告值（说明 CReFF 路径不会污染 I 帧路径）
- [ ] 同样权重 + gop_length=8 + 随机初始化 CReFFBlock，跑 inference，J&F 应**下降 5-10 个点**（说明 CReFF 起作用但还没训）

### 6.3 训练第一个 iter
- [ ] `loss.backward()` 后检查 `pixel_encoder.layer3[0].conv1.weight.grad` 不为 None 且不全 0
- [ ] 检查 `mask_decoder.pred.weight.grad` **应为 None**（验证冻结生效）
- [ ] 检查 `creff.lr_query_conv.weight.grad` 不为 None 且不全 0

### 6.4 训练 10 个 iter
- [ ] total_loss 在下降
- [ ] feat_distill loss 应从 ~0.1 量级开始
- [ ] 学习率显示正常（warmup 阶段在 0 → 1e-5 之间）

---

## 七、评测 schedule

| 时机 | 数据集 | 指标 | 目的 |
|---|---|---|---|
| 每 epoch | DAVIS-2017 val | J&F | 训练曲线监控 |
| 每 5 epoch | MOSEv1 val | J&F | 复杂场景验证 |
| Stage 2 结束 | DAVIS test-dev, MOSEv2 val, LVOS v2 val, YouTube-VOS 2019 val | J&F (+G for YT) | 主表 |
| Stage 2 结束 | DAVIS-2017 val (FPS) | FPS @ 720p, 1080p | 效率主张 |
| 论文撰写时 | LVOS v2 val | J&F vs 时距 d | 复刻 AR-Seg Fig 4(b) |

---

## 八、常见问题预案

| 现象 | 可能原因 | 应对 |
|---|---|---|
| Stage 1 J&F 跌很多（< 75） | LR 0.5× 太激进 | 改 0.7×，或冻 decoder 改为冻一半 |
| Stage 2 蒸馏 loss 不降 | feat_loss_weight 太小被淹没 | 调到 5.0，或先单独训 CReFF（冻 encoder） |
| Stage 2 J&F 不如 Stage 1 | CReFF 在 I 帧也插入了，污染 HR 路径 | 检查 `is_i_frame` 分流逻辑 |
| MOSEv2 上掉很多分 | 多目标 / 遮挡 / 小目标 MV 失效 | 加 fallback：MV confidence 低（解码端阈值）时退化为纯 attention |
| 长 video（LVOS）漂移 | HRPixelMemory 只 keep 2 个 GOP 太少 | 把 max_keep 调到 4，或加 LRU |
| OOM | seq_length 太长 | 降到 5；或开 gradient checkpointing |
| 训练发散（NaN） | AMP fp16 + warpFeature 在边界 NaN | warp 前后 dump tensor，确认 grid 在 [-1, 1] 内 |

---

## 九、目录结构建议

```
~/cutie-arseg/
├── pretrained/
│   └── cutie-base-mega.pth           # Stage 0 起点
├── data/
│   ├── davis2017/
│   │   ├── JPEGImages/Full-Resolution/...
│   │   └── compressed/3M-GOP12/{frames,MVmap_*}/...
│   ├── youtube-vos-2019/...
│   └── mose/...
├── exp/
│   ├── stage1_lr_adapt/
│   │   ├── checkpoints/
│   │   └── tb_log/
│   └── stage2_creff_fst/
│       ├── checkpoints/
│       └── tb_log/
└── code/                              # fork 的 cutie
    └── cutie/
```

---

## 九point五、当前实操配置与已知 gotcha（2026-05-31 起，权威版）

> **这一节是真正"在跑"的配置和踩过的坑。前面的两阶段 plan 是历史 plan，实际偏离很多，以本节为准。**

### 9.5.1 与原 plan 的偏离

| 项 | 原 plan | 实操 |
|---|---|---|
| Stage 1（LR 适配 encoder） | 30 epoch 单独训 encoder | **跳过**（待 step 500 J&F 出来后决定要不要补） |
| Stage 2 trainable | encoder + CReFFBlock 联合 | **CReFF-only**（encoder 冻死）。结论稳定，5/30 已确认联合训练会污染 encoder |
| 训练分辨率 | size=480（计划） | 早期 **size=128 默认值漏修**，已修正为 480 |
| seq_length | 8（计划） | 早期 **默认 3 漏修**，已修正为 8 |
| GOP-clip 关系 | 计划 gop=8 | 保持 gop=12，dataloader 加 clip_start 约束保证 clip 内必含 I 帧 |

### 9.5.2 数据配置（已确认：配置 A — 本机 HEVC compressed cache）

**没有外部 HEVC 数据集**，使用本机从 DAVIS 2017 原始帧预生成的 compressed cache：

```
data/DAVIS/2017/trainval/compressed/3M-GOP12/
├── frames/              24G    # HEVC 解码后的重构 RGB png
├── MVmap_GOP12_to_I/    85G    # libde265 导出的 MV bin（累计型，I→P_t）
├── hevc/                63M    # 中间 HEVC bitstream
└── manifest.json        1.6M
```

举例：
- `frames/bear/00000.png`
- `MVmap_GOP12_to_I/bear/00001.bin`

生成流程（见 `logs/davis_mv_full.nohup.log`）：
```
DAVIS jpg → ffmpeg → proxy.yuv
         → x265 --keyint 12 ... → proxy_3000.hevc
         → libde265/dec265 → decoded frames (png) + MV (bin)
```

**对应配置**：

| 配置 | RGB 来源 | MV 来源 | train-eval 对齐 |
|---|---|---|---|
| **A ✅ 当前** | HEVC 解码重构帧 | libde265 HEVC MV | ✅ 完全对齐 |
| B | 原始 DAVIS 2017 train JPEG | 同一视频预先 HEVC 编码后的 MV | ⚠️ RGB 分布不一致 |
| C | 原始 DAVIS JPEG | RAFT / 光流 | ⚠️ RGB + MV 都不一致 |

**结论**：训练读的是 HEVC 解码后的 frames + MV，eval 也走 HEVC 编码同一管线，**train-eval distribution 一致**，主表无需 gap 标注。dataset 类 `compressed_davis_dataset.py` 直接 load `frames/*.png` 和 `MVmap_GOP12_to_I/*.bin`，**不读** `JPEGImages/Full-Resolution`。

### 9.5.3 当前生效训练配置

```yaml
# 数据
size: 480                    # crop square 边长
seq_length: 8                # clip 长度（含至少 1 个 I 帧）
gop_length: 12               # 编码 GOP
lr_scale: 0.5                # LR 路径下采比例（HR 480 → LR 240）
num_objects: 3

# Crop / Resize 几何
resize_mode: short_side_to_size  # 短边 resize 到 480 后 random crop 480×480
crop_consistency: same_for_all_frames_in_clip  # 同 clip 8 帧共用一个 crop 坐标
clip_start_constraint: must_contain_i_frame    # 不再随机任意起点
mv_scale_with_resize: true   # MV 数值随 resize ×(480/orig_short_side)

# Trainable / Frozen
trainable: ["creff.*"]       # ~7,680 params
frozen: ["pixel_encoder", "pix_feat_proj", "mask_encoder", "key_proj",
         "pixel_fuser", "object_transformer", "object_summarizer", "mask_decoder"]

# Optimizer
optimizer: AdamW
lr_creff: 1e-4
weight_decay: 0.05
grad_clip: 0.5

# Batching
batch_size: 2                # 受 cgroup file cache 限制，b4 会 OOM
grad_accum: 16
effective_batch_size: 32

# 资源 / IO
amp: false                   # 训练侧不开 AMP（无 nan-skip 死循环）
num_workers: 1               # cgroup file cache 敏感，多 worker 反而吃满 cache
prefetch_factor: 1
pin_memory: false
persistent_workers: false
```

### 9.5.4 关键数据约定（必须和代码一致）

1. **MV 是累计型，不是帧间型**
   - `mv[t]` 表示 "I 帧 → P_t 的总位移"（pixel 单位）
   - t=0 时 MV=0，之后 |MV| 随 t 近似线性增长（drift-turn t=7 mean ≈ 156px @ 480 输入）
   - CReFF 单次 warp 直接用 `mv[t]`，**不要做差分**

2. **MV 符号约定：`+mv`（即 `grid = base + mv`）**
   - 等价于 "MV 表示 P 像素 → I 像素的采样偏移"（HEVC 标准约定）
   - 已用 `scripts/check_mv_sign_reconstruction.py` 验证过，三组样本 MSE(+mv) 都 << MSE(-mv)
   - 不要再做 `-mv` 实验

3. **MV 单位已经 /4**
   - libde265 raw MV 是 1/4-pel，loader 已经除过
   - **不要重复除**，否则 gold-fish 这类小运动序列 MV ≈ 1-2px，等于没运动

4. **MV resize 随空间一起缩放数值**
   - resize 时 `mv[..., 0] *= scale`，`mv[..., 1] *= scale`
   - crop 时只动 MV map 的空间排布，**不动 MV 数值**（位移是强度量）

### 9.5.5 已知 gotcha 总结

| # | 现象 | Root cause | 修复 |
|---|---|---|---|
| 1 | **silent crash 无 traceback** | cgroup file cache 撑爆 90GB 上限，被 oom_killer 杀（**不是** anon RSS，普通 RSS 监控看不到） | num_workers=1、posix_fadvise DONTNEED、预处理 dataset 到 540p；监控加 `cat /sys/fs/cgroup/.../memory.stat \| grep file` |
| 2 | **怀疑训练在 CPU 上**（false alarm） | watchdog 拉起 train_loop 后 RSS 飙到 50GB+ → 误以为退化到 CPU；实际 PID 821012 在 GPU 上，11.1GB VRAM、60% util | 不要从 RSS 推断 device；用 `nvidia-smi --query-compute-apps=pid,used_memory --format=csv` 直接确认 |
| 3 | **真退化到 CPU 的风险**（潜在） | watchdog 重启可能丢 env、`device='cpu'` 默认 | watchdog 脚本显式 `export CUDA_VISIBLE_DEVICES=0`；train 入口 assert `next(model.parameters()).is_cuda` |
| 4 | **default size=128** | 训练脚本 `--size` 默认值漏改，跑了第一批实验完全没意识到 | 明确写进 train_params；写进 ckpt 路径名；assert size >= 240 |
| 5 | **default seq_length=3** | 训练脚本 `--seq-length` 默认值漏改 | 同上；assert seq_length >= 5 |
| 6 | **训练 128 + eval 480 → J 远低于 baseline** | 分辨率 mismatch，pixel_encoder 看的输入分布完全不同 | 改 size=480 重训；eval 永远跟训练同分辨率 |
| 7 | **encoder+CReFF 联合训练 J 反而崩** | encoder 被 CReFF 信号污染，下游冻结模块（mask_encoder/decoder/transformer）对不上分布 | CReFF-only（只放开 `creff.*` 6 个张量、7680 params），encoder 冻死 |
| 8 | **联合训练后关掉 CReFF 也救不回 baseline** | 上一条的副作用：encoder 已经偏移，关 CReFF 也无济于事 | 出现污染就只能弃 ckpt 重训；不要寄希望"事后关 CReFF 补救" |
| 9 | **MV magnitude 看起来过大（drift-turn t=7 ≈ 156px）** | 累计型（I→P_t 总位移），在快运动序列上是物理合理的，不是 bug | 不要 clip / normalize；不要做帧间差分 |
| 10 | **`--disable-creff` 不是真正的 lower bound** | 当前实现把 P 帧也走 HR encode，等价于"no-CReFF + GOP1"，所以数字 == upper bound 0.8630 | 加新 eval mode："P 帧 LR encode 但不开 CReFF 注入"，~30 行 |
| 11 | **HR/LR 双分辨率训练能否对齐其他实验** | HR=fullsize、LR=480 的混合方案需要 dataloader 同时返回两套 RGB，evaluator 也要改 | 保持 HR=LR=480 单分辨率训练（lr_scale 在模型内部下采）；偏离对齐风险大于收益 |
| 12 | **random crop 让 MV 复杂** | crop 只改 MV map 的空间排布，**不改 MV 数值**（位移是强度量） | 8 帧 clip 共用一个 crop 坐标；MV bilinear sample 同 crop |
| 13 | **GOP=8 还是 GOP=12** | 计划文档写 GOP=8 让每个 clip 必含 I 帧；实际保持 GOP=12 + dataloader 加 `clip[0].is_i_frame` 约束 | 不动 GOP，改 clip start 采样策略 |
| 14 | **ckpt 间隔太密（25 step 一存）** | sanity 阶段为了密集观察设的，后续会让磁盘爆 | step 500 之后改成 100 或 250 一存 |
| 15 | **中间 ckpt 评测意义不大** | CReFF-only 100 step 就到 75.72，中间 200/300/400 单点信息量小 | 只在 500 / 1000 / 2000 / 早停时刻全集评测，省 GPU |

### 9.5.6 监控与韧性

```bash
# Resilient watchdog（崩了从最近 ckpt 续训）
PID 818821 / 821010

# Step-triggered eval watchers（ckpt 出来自动跑 full-val J&F）
step500  watcher PID 768356
step1000 watcher PID 768440

# 资源监控
logs/creff_s480_l8_b2a16_resource_trace.csv  # GPU util / VRAM / 主进程 RSS
```

写到 `output/validation_ledger.csv` 和 `.md`，每条记录包含：method / ckpt / train_step / train_params / eval_dataset / J / F / J&F。

### 9.5.7 Baseline 数字（决定"达标"含义）

| Reference | J | F | J&F | 来源 |
|---|---|---|---|---|
| **Upper bound**：Official Cutie + no-CReFF + 每帧 HR encode | 0.8243 | 0.9016 | **0.8630** | 2026-05-31 跑出 |
| **98% target** | — | — | **0.8457** | 0.8630 × 0.98 |
| **95% target**（可接受下限） | — | — | 0.8198 | 0.8630 × 0.95 |
| **Lower bound**：LR + no-CReFF + 每 P 帧 LR encode | — | — | TODO | 当前 `--disable-creff` 路径实际走 HR，需新加 eval mode |
| **CReFF-only step 500** (当前训练) | 0.6690 | 0.7446 | **0.7068** | 2026-05-31 watcher 落 ledger |
| CReFF-only step 1000 (当前训练) | TBD | TBD | TBD | watcher 待 step1000 ckpt |

注：`--disable-creff` 当前实现把 P 帧也走 HR encode，所以"no-CReFF + GOP12"实际 = "no-CReFF + GOP1"，两者数字一样（0.8630）。真正 lower bound 需要新代码路径："P 帧 LR encode 但不开 CReFF 注入"。

**step 500 解读**：0.7068 距 98% target 缺 0.139，距 95% 缺 0.113。属于决策树 "0.70-0.79" 档，root cause 大概率是 **LR 输入分布偏移**——encoder 是 HR 预训练，被强行喂 240×240 LR，CReFF 只能补一部分。下一步看 step 1000：
- step 1000 ≥ 0.75 且仍单调上升 → 让 CReFF 继续训到 plateau，再看 GOP sweep
- step 1000 在 0.70-0.72 平台 → 加 Stage 1（LR encoder adapt，30 epoch，lr=1e-5），然后再训 Stage 2 CReFF

### 9.5.8 本次会话 Q&A 沉淀（按时间序）

> 这一节是把 2026-05-30/31 两天和 assistant 来回讨论里**结论性**的问答归档。不是闲聊，是后续接手的人（包括未来的我）直接读的"为什么这么做"。

**Q1：full 验证用 480p 还是原始分辨率？**
- 训练用什么分辨率，eval 就用什么。当前 size=480，eval 也 480。混用必崩（gotcha #6）。

**Q2：训练时 HR 用 fullsize、LR 用 480，能不能这样混合？**
- 不能。dataloader 要返回两套 RGB，evaluator 也要改；混合方案偏离 AR-Seg / Cutie 标准实验设置，对齐风险大于收益。**保持单分辨率训练（HR=LR=480），lr_scale 在模型内部做下采**（gotcha #11）。

**Q3：seq_length 为什么是 8？**
- Cutie 主训默认值。短了 memory bank 没东西可读，长了 backprop 链太长、显存爆。8 是公认 sweet spot。

**Q4：GOP 也用 8 是不是更好？**
- 不用。GOP=8 是为了"每个 8 帧 clip 必含 I 帧"的便利；改 dataloader 的 clip start 采样策略（必须从 I 帧起）即可，不动 GOP=12（gotcha #13）。

**Q5：random crop 会让 MV 变复杂吗？**
- 不会。crop 只改 MV map 的空间排布，不改 MV 数值（位移是强度量）。同 clip 8 帧共用一个 crop 坐标即可（gotcha #12）。

**Q6：MV magnitude 看起来太大（t=7 ≈ 150px）是不是 bug？**
- 不是。MV 是累计型（I→P_t 总位移），快运动序列上是物理合理的。**不要 clip，不要 normalize，不要做帧间差分**（9.5.4 + gotcha #9）。

**Q7：MV 符号是 +mv 还是 -mv？**
- `+mv`（`grid = base + mv`）。`scripts/check_mv_sign_reconstruction.py` 三组样本验证完毕，MSE(+mv) 都远小于 MSE(-mv)。**不要再做 -mv 实验**（9.5.4 #2）。

**Q8：MV 要不要除以 4？**
- 不要。libde265 raw MV 是 1/4-pel，loader 已经除过了。重复除会让小运动序列（gold-fish）MV ≈ 1-2px，等于没运动（9.5.4 #3）。

**Q9：为什么要专门跑 no-CReFF baseline？**
- 因为要 anchor "CReFF 到底带来了多少收益"。upper bound（HR encode 每帧）= 0.8630 给出"如果你不压缩"的天花板；真正 lower bound（LR encode 每 P 帧 + 不开 CReFF）现在还缺，要新加 eval mode（gotcha #10）。

**Q10：训练为什么会无 traceback 中断？**
- cgroup file cache（不是 RSS）撑爆 90GB 上限被 oom_killer 杀。普通 `ps aux` 看不到，要 `cat /sys/fs/cgroup/.../memory.stat \| grep file`。修复：num_workers=1、posix_fadvise DONTNEED、把数据集预处理到 540p 减少 IO 量（gotcha #1）。

**Q11：训练是不是退化到 CPU 了？**
- 不是 false alarm。RSS 飙到 50GB+ 是 dataset/cache 占内存，跟 device 无关。`nvidia-smi --query-compute-apps` 直接看 GPU 占用最可靠（gotcha #2）。

**Q12：训练数据用的是 HEVC 数据集还是 raw DAVIS？**
- 都不是"外部数据集"。是本机用 ffmpeg + x265（GOP=12, 3 Mbps）+ libde265 从 DAVIS 2017 train JPEG 预处理出来的 cache：
  - `data/DAVIS/2017/trainval/compressed/3M-GOP12/frames/` （24G HEVC 重构 RGB）
  - `data/DAVIS/2017/trainval/compressed/3M-GOP12/MVmap_GOP12_to_I/` （85G libde265 MV）
  - 这就是 9.5.2 的**配置 A**：RGB 和 MV 来自同一条编码链，train-eval 完全对齐。

**Q13：训练 doc 不够详细，能不能补全？**
- 9.5 整节就是为这个开的。包含偏离、配置 A、yaml、MV 约定、gotcha 表、监控、baseline、本节 Q&A。

**Q14：踩过的坑、问过的问题都落进文档？**
- 本节（9.5.8）+ 9.5.5 gotcha 表扩到 15 条。后续每出现新坑，先加 gotcha 行，再视情况补到 Q&A。

### 9.5.9 实操约定（避免重复犯错）

1. **改默认值必写 assert**：`--size`、`--seq-length` 这种关键超参，入口处 `assert >= threshold`，不依赖记忆改默认。
2. **ckpt 路径名带超参**：`creff_s480_l8_b2a16_step500.pth`，路径名说话，避免事后看不出训练配置。
3. **eval 永远跟训练同分辨率**：train size=480 → eval size=480；train size=128 → 这个 ckpt 直接弃。
4. **encoder 一律冻死**（除非显式跑 Stage 1）：CReFF-only 是默认。
5. **中间 ckpt 不评全集**：只在 step 100（sanity）/ 500 / 1000 / 2000 / 早停时刻评。
6. **device 检查靠 nvidia-smi，不靠 RSS**。
7. **新加 loss 项前先消融**：feat_distill 之类的辅助 loss，加之前对照"原 Cutie loss 单独训"的 baseline，免得污染归因。

---

## 十、训练进展记录

> **本节由你（用户）报告进展时持续更新。每条记录格式：`日期 — 阶段 — epoch — 关键指标 — 笔记`**

### 进展日志

| 日期 | 实验配置 | step | DAVIS J&F | 备注 |
|---|---|---|---|---|
| 2026-05-30 | encoder+CReFF 联合训练 + gop=12 | 1000 | **54.85** 聚合，drift-chicane **8.81** | bimodal：慢序列 90+ / 快序列 <30；root cause 后确认是 encoder 被污染 |
| 2026-05-30 | 同上权重 + no-CReFF + gop=1 | 1000 | drift-chicane **65.53** | 关掉 CReFF 也救不回来 → encoder 污染坐实 |
| 2026-05-30 | **CReFF-only**（仅 7680 参数可训）+ gop=12 | 100 | drift-chicane **75.72** | 冻结 encoder 后 100 步就回血到 75 |
| 2026-05-30 | CReFF-only + gop=12 | 500 | _待 GPU 解锁_ | 单点 drift-chicane 已存 ckpt，等评测 |

参考上限：
- 官方权重 + no-CReFF + gop=1 (full HR Cutie)：drift-chicane **91.28**

### 当前最佳

- **CReFF-only step 100**：drift-chicane 75.72（gop=12）

### 已确认 OK 的设置

- **冻结策略**：只放开 `creff.*` 6 个参数张量（共 7680 params），冻结 `pixel_encoder` + `pix_feat_proj` + 所有 decoder/transformer/memory 模块
- **MV pipeline**：能在 100 步内让 CReFF 学到 75+，说明 warp_feature + MV 数据格式整体正确（如果错，CReFF 容量再大也学不到东西）
- **CReFF 模块结构**：256d × 7×7 局部 attention，对 stride-16 pix_feat 单点融合的设计有效

### 已 ruled out 的设置 / 踩过的坑

- **❌ encoder + CReFF 联合训练**（2026-05-30）
  - **后果**：encoder 被污染，即使关掉 CReFF 都救不回 baseline；快运动序列雪崩
  - **教训**：CReFF 是用 HR teacher 特征"注入信息"，encoder 同时被训练会去主动适配 CReFF 的注入分布，反而偏离原 Cutie 的 pix_feat 分布；下游冻结的 decoder/memory 全部对不上
  - **正确做法**：Stage 2 只训 CReFF，encoder 保持冻结。如果要训 encoder，必须先 Stage 1 单独训（不开 CReFF），让 encoder 适配 LR 输入后再进 Stage 2 仍然冻它

### Next checkpoints to confirm

- [ ] step 500 完整 DAVIS val 聚合 J&F（GPU 解锁后跑）
- [ ] 确认 blackswan/car-roundabout 等 90+ 序列没退化（验证 I 帧路径无副作用）
- [ ] GOP sweep（8/12/24/48）→ paper ablation + 暴露 MV pipeline 在大位移下是否还有残留 bug

---

## 一句话总结

**Stage 0 下载官方权重 → Stage 1 用 1e-5 微调 encoder 适应 LR（30 epoch）→ Stage 2 冻 decoder 套件、训 encoder + CReFFBlock、加 MSE 蒸馏（50 epoch）**。A100 ×4 共约 1.5 天，目标 J&F 保到 HR Cutie 的 98%，FPS 提升 2×。
