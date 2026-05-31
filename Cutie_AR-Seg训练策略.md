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

## 十、训练进展记录

> **本节由你（用户）报告进展时持续更新。每条记录格式：`日期 — 阶段 — epoch — 关键指标 — 笔记`**

### 进展日志

<!-- 空，等待第一条记录 -->

| 日期 | 阶段 | epoch | DAVIS J&F | MOSE J&F | 蒸馏 loss | 备注 |
|---|---|---|---|---|---|---|
| _待填_ | — | — | — | — | — | — |

### 当前最佳

- **Stage 1 最佳**：_未开始_
- **Stage 2 最佳**：_未开始_

### 已确认 OK 的设置

<!-- 你训练验证过的配置我会移到这里 -->

### 已 ruled out 的设置 / 踩过的坑

<!-- 你报告的失败实验我会归档到这里，附原因和教训 -->

---

## 一句话总结

**Stage 0 下载官方权重 → Stage 1 用 1e-5 微调 encoder 适应 LR（30 epoch）→ Stage 2 冻 decoder 套件、训 encoder + CReFFBlock、加 MSE 蒸馏（50 epoch）**。A100 ×4 共约 1.5 天，目标 J&F 保到 HR Cutie 的 98%，FPS 提升 2×。
