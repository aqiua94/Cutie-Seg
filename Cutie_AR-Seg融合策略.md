# Cutie + AR-Seg 融合策略（基于源码级分析）

> 基于阅读 AR-Seg (`/tmp/ar-seg`) 和 Cutie (`/tmp/cutie`) 完整源码后整理。
> 目标：把 AR-Seg 的 CReFF + FST + Altering Resolution 三件套，移植到 Cutie 的 VOS 流水线上，给出 code-level 的对接点、模块改造方案和训练-推理流程。

---

## 一、AR-Seg 源码结构速查

### 1.1 关键文件与作用

| 文件 | 关键类/函数 | 作用 |
|---|---|---|
| `model/pspnet.py` | `PSPNetWithFuse` (默认 V1) | 主模型，含 `forward_phase1`/`forward_phase2` 双阶段接口 |
| `model/pspnet.py` | `PSPNetWithFuseV2/V3` | CReFF 插在 PSP 输入 (512d) 或 stem 之后 (64d at 1/4) 的变体 |
| `model/attention.py` | `MyAttention` (默认 `'local'`) | CReFF 的 ℱ_LA，7×7 局部 attention，依赖 CUDA op `f_similar / f_weighting` |
| `evaluation.py` | `warpFeature(feature, flow)` | MV 对特征的 grid_sample warp |
| `evaluation.py` | `EvalAlterRes` | 双分辨率 + MV 推理评测器 |
| `train_pair.py` | `train()` | 含 stage1（无融合）/stage2（融合 + MSE 蒸馏）的完整训练 |
| `dataset/camvid.py` | `CamVidWithFlow` | 同时返回 `(img, label, y_cls, ref_img, flow)` |
| `pre-process/generate_compressed_dataset_camvid.py` | — | HEVC 编码 + MV 导出 (.bin, int16/4) |

### 1.2 CReFF 核心流程（PSPNetWithFuse, V1, mid_dim=64）

```python
# forward_phase1：仅运行 backbone+PSP+3 次 PSPUpsample，得到 1/2 分辨率、64 通道的 p
def forward_phase1(self, x):
    f, class_f = self.feats(x)         # ResNet18 → f(stride 32, 512d), class_f(stride 16)
    p = self.psp(f); p = self.drop_1(p)            # 1024d, stride 32
    p = self.up_1(p); p = self.drop_2(p)            # 256d, stride 16
    p = self.up_2(p); p = self.drop_2(p)            # 64d,  stride 8
    p = self.up_3(p); p = self.drop_2(p)            # 64d,  stride 4
    return self.classifier(auxiliary), p

# forward_phase2：用 HR ref 特征 + LR 当前特征做 CReFF，再走分类头
def forward_phase2(self, p, ref_p):
    p = self.fuse_attention(ref_p, p)              # MyAttention(hr_feat=ref_p, lr_feat=p)
    out = self.final_conv(p)
    out = F.interpolate(out, (H, W), mode='bilinear', align_corners=True)
    return self.final_logsoftmax(out), p
```

### 1.3 MyAttention（CReFF 的 ℱ_LA）核心

```python
class MyAttention(nn.Module):                       # attention.py:157
    def __init__(self, feat_dim, kW, kH):
        self.lr_query_conv = Conv2d(C, C, 3, 1, groups=C)   # depthwise
        self.hr_key_conv   = Conv2d(C, C, 3, 1, groups=C)
        self.hr_value_conv = Conv2d(C, C, 3, 1, groups=C)
        self.softmax = nn.Softmax(dim=3)
        self.kH, self.kW = kH, kW                            # 默认 7

    def forward(self, hr_feat, lr_feat):
        N, C, H, W = hr_feat.shape
        lr_feat = F.interpolate(lr_feat, (H, W), mode='bilinear')   # 先把 LR 上采到 HR 大小
        Q = self.lr_query_conv(lr_feat)
        K = self.hr_key_conv(hr_feat)
        V = self.hr_value_conv(hr_feat)
        w = f_similar(Q, K, kH, kW)                          # CUDA op: 每点 vs 7×7 邻域 → [N,H,W,49]
        w = self.softmax(w)
        out = f_weighting(V, w, kH, kW)                      # CUDA op: 加权和 V 的 7×7 邻域
        return lr_feat + out                                 # 残差
```

`f_similar` / `f_weighting` 来自 `localAttention` 包（github: zzd1992/Image-Local-Attention），是一对编译过的 CUDA kernel，比纯 PyTorch unfold 快约 10×。

### 1.4 warpFeature（MV 用作 flow 做 grid_sample）

```python
def warpFeature(feature, flow):                     # evaluation.py:61
    B, C, H, W = feature.shape
    flow = flow.permute(0, 3, 1, 2)                          # [B,2,H,W]
    xx, yy = meshgrid(W, H); grid = stack([xx, yy], 1).float().cuda()
    vgrid = grid + flow                                       # 像素坐标 + 位移
    vgrid[:,0] = 2.0 * vgrid[:,0] / (W-1) - 1.0
    vgrid[:,1] = 2.0 * vgrid[:,1] / (H-1) - 1.0
    return F.grid_sample(feature, vgrid.permute(0,2,3,1))
```

调用前对 MV 做了：
1. 通道维转到第二维 (`transpose(2,3).transpose(1,2)`)
2. 按特征图尺寸缩放：`flow * highres_ref_p.shape[-2] / flow.shape[-2]`（注意只缩 H 不缩 W，AR-Seg 默认 4:3 比例 720×960 所以无 bug；用到 16:9 数据集时要分别缩）
3. `F.interpolate(... mode='bilinear')` 把 MV 下采样到特征分辨率
4. 再转回 `[B,H,W,2]` 喂 warpFeature

### 1.5 训练 (`train_pair.py`)

关键三段：

```python
# (a) HR teacher 是冻结的，运行 HR 输入得到 highres_p（蒸馏目标）和 highres_ref_p（参考帧 HR 特征）
with torch.no_grad():
    _, _, highres_p     = highres_net(x_HR)
    if epoch >= stage1_epoch:
        _, _, highres_ref_p = highres_net(ref_x_HR)

# (b) 把 MV 缩放到特征分辨率后 warp 参考帧 HR 特征
flow = (flow * highres_ref_p.shape[-2] / flow.shape[-2])
flow = F.interpolate(flow, [highres_ref_p.shape[-2], highres_ref_p.shape[-1]], mode='nearest')
highres_ref_p = warpFeature(highres_ref_p, flow)

# (c) Student（LR）走 phase1+phase2 融合，损失 = seg_loss + α·cls_loss + λ·MSE(highres_p, out_p)
out, out_cls, out_p = net(x_LR, mode='merge', ref_p=highres_ref_p)
loss = seg_criterion(out, y) + α * cls_criterion(out_cls, y_cls)
if feat_loss == 'mse':
    loss = loss + nn.MSELoss()(highres_p, out_p)
```

附加：`load_decoder()` 把 HR 训好的 `final_conv` 权重直接拷到 student，并 `requires_grad=False`（FST 的隐式共享 decoder 约束）。

### 1.6 数据流（CamVidWithFlow）

```python
return img, label, existence_list, ref_img, flow_map
# flow_map 来自 .bin 文件：np.fromfile(f, np.short).reshape(720, 960, 2) / 4
```

预处理脚本要点（`pre-process/generate_compressed_dataset_camvid.py`）：
- 用 x265 编码，控制 GOP=`ref_gap`、bitrate=3Mbps
- 用魔改 libde265 把每帧的 MV 导出为 `.bin`（int16，单位 1/4 像素）
- 文件命名包含距离索引：`MVmap_GOP{G}_dist_{d}/<seq>/<frame>.bin`

---

## 二、Cutie 源码结构速查

### 2.1 关键文件与作用

| 文件 | 关键类 | 作用 |
|---|---|---|
| `cutie/model/cutie.py` | `CUTIE` | 主模型组装；含 `encode_image / encode_mask / transform_key / read_memory / pixel_fusion / readout_query / segment` |
| `cutie/model/big_modules.py` | `PixelEncoder` | ResNet-50 输出多尺度 `(f16, f8, f4)` |
| `cutie/model/big_modules.py` | `KeyProjection` | f16 → (key, shrinkage, selection) for memory match |
| `cutie/model/big_modules.py` | `MaskEncoder` | ResNet-18 把 prev mask+image 编为 `msk_value` (stride 16) |
| `cutie/model/big_modules.py` | `MaskDecoder` | f16 + memory readout → 1/8 → 1/4 logits |
| `cutie/model/transformer/object_transformer.py` | `QueryTransformer` | object-level 推理 |
| `cutie/model/train_wrapper.py` | `CutieTrainWrapper.forward` | 训练循环：跑 seq_length 帧，逐帧 encode_mask → read_memory → segment |
| `cutie/model/trainer.py` | `Trainer` | 优化器/AMP/loss 调度 |
| `cutie/model/losses.py` | `LossComputer` | bootstrapped CE + Dice + aux loss |
| `cutie/inference/inference_core.py` | `InferenceCore.step` | 推理时单帧入口，决定写不写 memory |
| `cutie/inference/memory_manager.py` | `MemoryManager` | 工作/长时/sensory memory 管理 |
| `cutie/inference/image_feature_store.py` | `ImageFeatureStore` | 缓存 `(ms_features, pix_feat, key, shrinkage, selection)` |
| `cutie/config/model/base.yaml` | — | pixel_dim=256, key_dim=64, value_dim=256, sensory_dim=256, embed_dim=256, ms_dims=[1024,512,256] |

### 2.2 Cutie 的 "pixel feature" 通路（CReFF 的天然插入点）

```python
def encode_image(self, image):                       # cutie.py:61
    image = (image - self.pixel_mean) / self.pixel_std
    ms_image_feat = self.pixel_encoder(image)        # 返回 (f16, f8, f4)
    return ms_image_feat, self.pix_feat_proj(ms_image_feat[0])
    #                                       ^^^^^^^^^^^^^^^^
    #                                       1x1 conv: f16 (1024d) → pix_feat (256d, stride 16)
```

下游分两路：
1. `pix_feat (256d, 1/16)` → `read_memory()` → memory readout → `pixel_fusion()` → `object_transformer` → `segment()`
2. `ms_image_feat[0]=f16` → `transform_key()` → key/shrinkage/selection（写入 memory）

> **关键洞察**：`pix_feat` 是 stride-16、256d 的单一稠密特征，正是 AR-Seg `PSPNetWithFuse` 里 `p` 的对应物。所有下游模块（key/value memory、transformer、decoder）都吃这一个 tensor。**在它前面插 CReFF，就把双分辨率改造的影响完全控制在 encoder 出口一处。**

### 2.3 Cutie 的 Memory 通路

```python
# 推理时（InferenceCore._add_memory）
self.memory.initialize_sensory_if_needed(key, all_obj_ids)
msk_value, sensory, obj_value, _ = self.network.encode_mask(image, pix_feat, sensory, prob, ...)
self.memory.add_memory(key, shrinkage, msk_value, obj_value, all_obj_ids,
                       selection=selection, as_permanent=as_permanent)

# 关键参数：cfg.mem_every（默认 5），第一帧永远 permanent，之后每 mem_every 帧写一次
is_mem_frame = ((self.curr_ti - self.last_mem_ti >= self.mem_every) or (mask is not None)) and (not end)
```

Memory 存的是 mask-conditioned value，**和我们要存的 "HR I-frame pix_feat" 不是同一个东西**——所以需要新加一个并行的 bank（见第三节）。

---

## 三、模块映射表（AR-Seg ↔ Cutie）

| AR-Seg 元素 | Cutie 中的对应物 / 改造方案 |
|---|---|
| **HR teacher network** (`highres_net`, 冻结的 PSPNet) | 复用同一份 Cutie 权重：teacher 跑全分辨率，student 共享参数但走 LR 输入。或者更省显存：teacher = student（同一个网络），只是在 I 帧时 forward 完整分辨率并把 `pix_feat` cache 起来。**推荐方案：单网络双分辨率**（FST 的"共享 decoder"自然达成） |
| **`forward_phase1`** (extract LR feature) | `CUTIE.encode_image(image_LR)` |
| **`forward_phase2`** (CReFF + decoder head) | 新增 `CUTIE.encode_image_with_creff(image_LR, ref_pix_feat_HR, mv)`：encode_image → MV warp(ref_pix_feat) → MyAttention 融合 → 替换 `pix_feat`，下游不变 |
| **`MyAttention` (mid_dim=64, k=7)** | **`CReFFBlock(feat_dim=256, kH=7, kW=7)`**，结构完全复用 AR-Seg 的 `attention.py` `MyAttention`，仅把 feat_dim 从 64 改 256 |
| **`warpFeature`** | 原样拿过来（`F.grid_sample` 实现），独立一个 `mv_warp.py` |
| **HR keyframe feature bank** | 新增 `HRPixelMemory`：一个 dict `{gop_idx: pix_feat_HR}`，最多保留近 N 个 GOP（推荐 N=2，即上一个 + 当前 I 帧）；和 `MemoryManager` 平行存在 |
| **`final_conv` 冻结** | 冻结 Cutie 的 `mask_decoder + key_proj + pixel_fuser + object_transformer + object_summarizer + mask_encoder` 一整套（FST 隐式约束的关键） |
| **MSE feat loss** (`feat_criterion(highres_p, out_p)`) | `MSELoss(pix_feat_HR_teacher, pix_feat_LR_after_creff)` —— **HR teacher 直接 forward 完整分辨率得到的 `pix_feat`** 是蒸馏目标 |
| **`CamVidWithFlow` 数据集** | 改造 Cutie 的 `VOSDataset`（`cutie/dataset/`）为 `VOSDatasetCompressed`：每条样本除了 RGB+mask 序列，还携带每帧的 GOP 类型（I/P）、参考帧索引、MV map |
| **HEVC 预处理脚本** | 直接复用 `pre-process/generate_compressed_dataset_camvid.py` 的思路，对 DAVIS/YouTube-VOS/MOSE 视频重新编码并导出 MV |

---

## 四、新增/改动的代码模块

### 4.1 `cutie/model/creff.py`（新建）

```python
# Copy MyAttention from ar-seg/model/attention.py
# 关键改动：feat_dim 从 64 → 256；保留 7x7 邻域；保留 localAttention CUDA op
from localAttention import similar_forward, similar_backward, weighting_forward, ...

class CReFFBlock(nn.Module):
    def __init__(self, feat_dim=256, kH=7, kW=7):
        super().__init__()
        self.lr_query_conv = nn.Conv2d(feat_dim, feat_dim, 3, 1, 1, groups=feat_dim)
        self.hr_key_conv   = nn.Conv2d(feat_dim, feat_dim, 3, 1, 1, groups=feat_dim)
        self.hr_value_conv = nn.Conv2d(feat_dim, feat_dim, 3, 1, 1, groups=feat_dim)
        self.softmax = nn.Softmax(dim=3)
        self.kH, self.kW = kH, kW
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if ly.bias is not None: nn.init.constant_(ly.bias, 0)

    def forward(self, hr_feat, lr_feat):
        # hr_feat: [B, C, H, W]   (MV-warped HR I-frame pix_feat at HR stride-16)
        # lr_feat: [B, C, h, w]   (current P-frame pix_feat at LR stride-16)
        N, C, H, W = hr_feat.shape
        lr_feat = F.interpolate(lr_feat, (H, W), mode='bilinear', align_corners=True)
        Q = self.lr_query_conv(lr_feat)
        K = self.hr_key_conv(hr_feat)
        V = self.hr_value_conv(hr_feat)
        w = f_similar(Q, K, self.kH, self.kW)
        w = self.softmax(w)
        out = f_weighting(V, w, self.kH, self.kW)
        return lr_feat + out
```

### 4.2 `cutie/model/mv_warp.py`（新建）

```python
def warp_feature(feature, flow):
    """
    feature: [B, C, H_f, W_f]
    flow:    [B, H_mv, W_mv, 2]   单位：像素位移
    返回：    [B, C, H_f, W_f]
    """
    B, C, H, W = feature.shape
    # 1. 缩放 MV 到特征分辨率（分别缩 H/W）
    flow = flow.permute(0, 3, 1, 2)                          # [B,2,H_mv,W_mv]
    sy, sx = H / flow.shape[2], W / flow.shape[3]
    flow[:, 0] *= sx
    flow[:, 1] *= sy
    flow = F.interpolate(flow, (H, W), mode='bilinear', align_corners=False)
    # 2. 构 grid 并归一化
    xx = torch.arange(W, device=feature.device).view(1,1,1,W).expand(B,1,H,W)
    yy = torch.arange(H, device=feature.device).view(1,1,H,1).expand(B,1,H,W)
    grid = torch.cat([xx, yy], 1).float() + flow
    grid[:, 0] = 2.0 * grid[:, 0] / max(W - 1, 1) - 1.0
    grid[:, 1] = 2.0 * grid[:, 1] / max(H - 1, 1) - 1.0
    return F.grid_sample(feature, grid.permute(0, 2, 3, 1), align_corners=True)
```

> 修正了 AR-Seg 源版本的 bug：原版只按 H 缩 flow 不缩 W，对 16:9 数据集（DAVIS 1080×1920、MOSE 等）会算错。

### 4.3 `cutie/model/cutie.py` 改动

在 `CUTIE.__init__` 加：

```python
self.use_creff   = model_cfg.get('use_creff', False)
self.creff_k     = model_cfg.get('creff_k', 7)
if self.use_creff:
    self.creff = CReFFBlock(feat_dim=self.pixel_dim, kH=self.creff_k, kW=self.creff_k)
```

新增方法（不改原 `encode_image`，便于消融）：

```python
def encode_image_compressed(self, image_LR, ref_pix_feat_HR=None, mv=None):
    """
    image_LR:        [B, 3, H/α, W/α]   下采样过的当前帧
    ref_pix_feat_HR: [B, C, H/16, W/16] 上一个 I 帧的 HR pix_feat（已 cache）
    mv:              [B, H_mv, W_mv, 2] 当前帧 → ref 帧的运动矢量（HR 像素单位）

    返回 (ms_feat_LR, pix_feat_fused)；pix_feat_fused 已经升到 HR stride-16
    """
    ms_feat_LR, pix_feat_LR = self.encode_image(image_LR)
    if ref_pix_feat_HR is None or mv is None or not self.use_creff:
        return ms_feat_LR, pix_feat_LR                    # 退化为标准路径（I 帧）
    warped_ref = warp_feature(ref_pix_feat_HR, mv)
    pix_feat_fused = self.creff(warped_ref, pix_feat_LR)  # CReFF
    return ms_feat_LR, pix_feat_fused
```

> 关键细节：`ms_feat_LR` 里的 `f8`、`f4` 是 LR 分辨率的，会进入 `MaskDecoder` 做 skip-connection。两种处理：
> (a) 简单：直接 bilinear 上采到 HR 尺寸再喂 decoder（信息略糊但实现 0 改动）；
> (b) 复杂：CReFF 同时对 f8、f4 做一次（要付出额外 attention 计算）。
> **推荐 (a) 起步**，与 AR-Seg 论文中"CReFF 只在 mid-level 单点插入"的设定一致；ablation 时再试 (b)。

### 4.4 `cutie/inference/hr_pixel_memory.py`（新建）

```python
class HRPixelMemory:
    """与 MemoryManager 平行，仅存 I 帧 pix_feat。"""
    def __init__(self, max_keep: int = 2):
        self.max_keep = max_keep
        self.store = {}          # {gop_idx: (ti, pix_feat_HR)}
        self.last_gop_idx = -1

    def add(self, gop_idx, ti, pix_feat_HR):
        self.store[gop_idx] = (ti, pix_feat_HR)
        self.last_gop_idx = gop_idx
        # 淘汰最老
        while len(self.store) > self.max_keep:
            oldest = min(self.store.keys())
            del self.store[oldest]

    def get_latest(self):
        return self.store[self.last_gop_idx][1]
```

### 4.5 `cutie/inference/inference_core.py` 改动

```python
class InferenceCore:
    def __init__(self, network, cfg, ...):
        ...
        self.gop_length     = cfg.get('gop_length', 12)
        self.lr_scale       = cfg.get('lr_scale', 0.5)
        self.use_creff      = cfg.get('use_creff', False)
        self.hr_pix_memory  = HRPixelMemory(max_keep=2) if self.use_creff else None

    def step(self, image, mask=None, objects=None, *, mv=None, ...):
        self.curr_ti += 1
        is_i_frame = (self.curr_ti % self.gop_length == 0) or (mask is not None)

        if not self.use_creff or is_i_frame:
            # I 帧：标准 Cutie 流程，HR 输入
            ms_feat, pix_feat = self.image_feature_store.get_features(self.curr_ti, image)
            key, shr, sel = self.image_feature_store.get_key(self.curr_ti, image)
            if is_i_frame and self.use_creff:
                self.hr_pix_memory.add(self.curr_ti // self.gop_length, self.curr_ti, pix_feat)
        else:
            # P 帧：LR 输入 + CReFF
            image_LR = F.interpolate(image.unsqueeze(0), scale_factor=self.lr_scale,
                                     mode='bilinear', align_corners=False)[0]
            ref_pix_feat_HR = self.hr_pix_memory.get_latest()
            ms_feat, pix_feat = self.network.encode_image_compressed(
                image_LR.unsqueeze(0), ref_pix_feat_HR, mv.unsqueeze(0))
            # key/shrinkage/selection 由融合后的 pix_feat 重算（保持 memory match 质量）
            from cutie.model.big_modules import KeyProjection  # 已是网络一部分
            # 等价于 transform_key(ms_feat[0])，但我们已经替换了 pix_feat，需要走 KeyProjection on f16
            # 这里偷个懒，复用 transform_key(ms_feat[0])（仍然是 LR f16）。如果对 memory match 影响大，
            # 改成 transform_key on upsampled f16 或者直接 cache 上次的 key。
            key, shr, sel = self.network.transform_key(ms_feat[0])

        # 下游完全不变
        ...
```

### 4.6 `cutie/model/train_wrapper.py` 改动

```python
class CutieTrainWrapper(CUTIE):
    def __init__(self, cfg, stage_cfg):
        super().__init__(cfg, single_object=(stage_cfg.num_objects == 1))
        self.lr_scale     = stage_cfg.get('lr_scale', 0.5)
        self.gop_length   = stage_cfg.get('gop_length', 8)
        self.feat_loss_w  = stage_cfg.get('feat_loss_weight', 1.0)
        self.use_creff    = cfg.model.get('use_creff', False)
        self.freeze_decoder_for_fst = stage_cfg.get('freeze_decoder', True)
        if self.freeze_decoder_for_fst and self.use_creff:
            for m in [self.mask_decoder, self.key_proj, self.pixel_fuser,
                      self.mask_encoder, self.object_transformer, self.object_summarizer]:
                for p in m.parameters():
                    p.requires_grad = False

    def forward(self, data):
        out = {}
        frames    = data['rgb']                        # [B, T, 3, H, W]
        first_gt  = data['first_frame_gt'].float()
        mv_seq    = data.get('mv', None)               # [B, T, H_mv, W_mv, 2]，I 帧位置可为零
        is_i      = data.get('is_i_frame', None)       # [B, T] bool；first frame 必须为 True
        b, T = frames.shape[:2]

        # 1. 先按 is_i 把每帧分流：I 帧走 HR 路径并 cache pix_feat（同时作为 teacher target）
        ms_feat_all, pix_feat_all = [], []
        teacher_pix_feat_all = []                       # HR teacher target，用于 MSE
        ref_pix_feat_HR = None                          # 当前 GOP 的 HR I-frame pix_feat
        for t in range(T):
            if is_i[:, t].all():
                ms_t, pix_t = self.encode_image(frames[:, t])             # HR
                ref_pix_feat_HR = pix_t                                   # cache
                teacher_pix_feat_all.append(pix_t.detach())               # teacher 取自身（HR）
            else:
                with torch.no_grad():
                    _, teacher_pix_t = self.encode_image(frames[:, t])    # HR teacher forward
                teacher_pix_feat_all.append(teacher_pix_t)

                img_lr = F.interpolate(frames[:, t], scale_factor=self.lr_scale, mode='bilinear')
                # 注意：把 ref 帧 HR pix_feat 用 MV warp
                warped = warp_feature(ref_pix_feat_HR, mv_seq[:, t])
                ms_t, pix_lr = self.encode_image(img_lr)
                pix_t = self.creff(warped, pix_lr)
                # ms_t 的 f8/f4 是 LR 的，上采到 HR
                ms_t = [F.interpolate(f, scale_factor=1/self.lr_scale, mode='bilinear') for f in ms_t]
            ms_feat_all.append(ms_t)
            pix_feat_all.append(pix_t)

        # 2. 后续逻辑（key/shrinkage/selection、memory 读写、segment）和原 forward 完全一样
        # ...

        # 3. 增加蒸馏损失
        out['feat_distill_pairs'] = list(zip(teacher_pix_feat_all, pix_feat_all))   # 交给 LossComputer
        return out
```

`LossComputer` 增加：

```python
# cutie/model/losses.py
if 'feat_distill_pairs' in out:
    feat_loss = 0
    for teacher, student in out['feat_distill_pairs']:
        feat_loss = feat_loss + F.mse_loss(student, teacher)
    losses['feat_distill'] = feat_loss / len(out['feat_distill_pairs'])
    losses['total_loss']   = losses['total_loss'] + self.feat_loss_w * losses['feat_distill']
```

---

## 五、训练流程（两阶段 + FST，对应 AR-Seg 的 stage1/stage2）

### Stage 0 — HR baseline 训练
- 用 Cutie 官方四阶段流程（pretrain on static images → main training on YT-VOS+DAVIS+MOSE → 等）训完 HR Cutie；这就是 teacher，也是 student 的初始化。
- 来源：直接下载官方权重 `cutie-base-mega.pth`。

### Stage 1 — LR 适配（无融合）
- 输入：LR 帧（`scale=0.5`）
- 网络：Cutie 全部参数可训（但 `mask_decoder + transformer` 通常冻结，因为来自 HR 训练）
- Loss：原版 Cutie 的 bootstrapped CE + Dice + aux
- 目的：让 pixel_encoder 适应 LR 输入；与 AR-Seg `stage1_epoch` 之前段对应
- Epochs：建议 30-50

### Stage 2 — 双分辨率 + CReFF（FST）
- 数据：(I 帧 HR, P 帧 LR + MV)
- 网络：
  - `pixel_encoder` 可训
  - `CReFFBlock` 可训
  - `mask_encoder, key_proj, pixel_fuser, object_transformer, object_summarizer, mask_decoder` 全部冻结（隐式共享 decoder）
- HR teacher：用同一个 `pixel_encoder` 但跑 HR 输入并 `torch.no_grad()`，得到 `pix_feat_HR_teacher`
- Loss：原 Cutie loss + λ · MSE(`pix_feat_HR_teacher`, `pix_feat_LR_after_creff`)
- λ 推荐 1.0（对齐 AR-Seg 默认）
- Epochs：50-100

### 训练 trick（来自 AR-Seg）
- `train_pair.py:265-279`：snapshot 存在时用 `GradualWarmupScheduler` warmup 500 steps，无 snapshot 时直接 CosineAnnealing；移植时建议沿用
- `train_pair.py:209-225`：用 cityscapes 时换 SGD + momentum 0.9，长 video benchmark 用 AdamW 即可，沿用 Cutie 默认

---

## 六、推理流程

### 6.1 GOP 调度

- 在线模式：固定 GOP 长度 `L=12`（与 AR-Seg 默认一致），每 L 帧一个 I 帧
- 离线模式（推荐评测时用）：直接读 HEVC 比特流的 I/P 标签，I=keyframe，P=用 MV 补偿
- 第一帧（带 GT mask）强制为 I 帧（permanent memory）

### 6.2 单帧处理伪码

```python
for ti, (frame_BGR, mv_or_None) in enumerate(video):
    is_i = (ti % gop_length == 0) or first_frame
    if is_i:
        # HR 路径
        ms_feat, pix_feat = network.encode_image(frame_HR)
        hr_pix_memory.add(ti // gop_length, ti, pix_feat)
        key, shr, sel = network.transform_key(ms_feat[0])
    else:
        # LR + CReFF 路径
        frame_LR = downsample(frame_HR, lr_scale)
        ref_HR   = hr_pix_memory.get_latest()
        ms_feat, pix_feat = network.encode_image_compressed(frame_LR, ref_HR, mv_or_None)
        key, shr, sel = network.transform_key(ms_feat[0])
    # 下游：read_memory → object_transformer → segment → 返回 mask
    out_prob = inference_core.continue_step(pix_feat, ms_feat, key, shr, sel, ...)
```

### 6.3 与 Cutie Memory 的协同

- `MemoryManager.mem_every`：保持原值 5（与 GOP 解耦，让两套 memory 互不打架）
- I 帧时双重缓存：既存到 `MemoryManager`（mask-conditioned value）又存到 `HRPixelMemory`（pix_feat）
- P 帧时不写 `HRPixelMemory`，但 `MemoryManager` 仍按原规则更新（用 CReFF 后的 pix_feat）

---

## 七、数据流水线

### 7.1 预处理

1. **下载视频原始文件**：DAVIS 提供 1080p 原始 MP4；MOSEv1/v2 直接提供视频；YouTube-VOS 需从其 frames 重新拼接（已无原码流）
2. **HEVC 编码**：复用 `pre-process/x265/` 工具链
   ```bash
   ffmpeg -framerate 24 -i frames/%05d.png -c:v libx265 -x265-params "keyint=12:bitrate=3000" out.h265
   ```
3. **解码 + MV 导出**：复用 AR-Seg 魔改的 `libde265`，每帧导出 `.bin`（int16 H×W×2，单位 1/4 像素）
   - 对每个 GOP 内的 P 帧，导出"P → I"的 MV
4. 目录结构推荐：
   ```
   davis2017/
   ├── JPEGImages/480p/<seq>/<frame>.jpg     (原始)
   ├── compressed/3M-GOP12/frames/<seq>/...  (HEVC 重建帧)
   └── compressed/3M-GOP12/MVmap_dist_*/<seq>/<frame>.bin
   ```

### 7.2 Dataset 改造

`cutie/dataset/vos_dataset.py` → 新版 `vos_dataset_compressed.py`：

```python
def __getitem__(self, idx):
    seq_name, frame_ids = self._sample_clip(idx)
    frames, masks, mvs, is_i = [], [], [], []
    for k, fid in enumerate(frame_ids):
        is_i_k = (fid % self.gop_length == 0) or (k == 0)
        if is_i_k:
            img = load(f'compressed/.../frames/{seq_name}/{fid}.jpg')
            mv  = torch.zeros(self.H, self.W, 2)             # I 帧 MV 占位
        else:
            img = load(f'compressed/.../frames/{seq_name}/{fid}.jpg')
            mv  = load_bin(f'compressed/.../MVmap.../{seq_name}/{fid}.bin')
            mv  = torch.from_numpy(mv).float() / 4.0          # int16 1/4 像素 → float pixel
        frames.append(img); masks.append(load_mask(...))
        mvs.append(mv); is_i.append(is_i_k)
    return {
        'rgb': stack(frames), 'first_frame_gt': masks[0],
        'mv': stack(mvs), 'is_i_frame': torch.tensor(is_i),
        ...
    }
```

---

## 八、与 Cutie 原架构的兼容性检查

| 模块 | 是否需要改 | 风险点 |
|---|---|---|
| `PixelEncoder` (ResNet-50) | 否 | 直接吃 LR 输入即可 |
| `pix_feat_proj` (1×1 conv) | 否 | 通道一致 |
| `KeyProjection` | 否，但要决定喂哪个 f16 | 用 LR f16 → memory match 略糊；用上采后 f16 → 更准但稍贵。建议起步用 LR f16，做 ablation |
| `MaskEncoder` | 否，**冻结** | Stage 2 不更新 |
| `pixel_fuser` | 否，**冻结** | 它吃的是 `pix_feat`（已经融合到 HR 分辨率），完全兼容 |
| `MemoryManager` | 否 | 与 `HRPixelMemory` 平行存在 |
| `MaskDecoder` | 否，**冻结** | 它需要 f8、f4 做 skip；P 帧时是 LR 上采的，会略糊 → CReFF 蒸馏要把这部分误差也压下去 |
| `ObjectTransformer` | 否，**冻结** | 吃的是 pixel_readout（B×N×C×H×W），兼容 |
| `ImageFeatureStore` | 改 | 加 `gop_idx` 维度，区分 HR/LR 缓存 |

---

## 九、与 AR-Seg 不同的设计取舍

| 维度 | AR-Seg | Cutie 移植版 |
|---|---|---|
| Teacher 网络 | 单独训的 HR PSPNet，冻结 | **同一个 Cutie，仅运行 HR 输入** —— 省一份参数，FST 隐式约束更强 |
| CReFF 通道数 | 64（PSP 解码后） | **256**（Cutie pixel_dim）—— attention 内存增 16×，但 stride-16 分辨率小，仍 OK |
| CReFF 插入点 | up_3 之后（stride 4） | **pix_feat_proj 之后（stride 16）** —— 计算量更小，且 stride 16 是所有下游模块共享起点 |
| MV warp 对象 | mid-level 64d 特征 | 256d pix_feat |
| 蒸馏目标 | HR teacher 的 stride-4 特征 | HR teacher 的 stride-16 pix_feat |
| 数据范式 | 单帧 + 1 个 ref 帧 | 视频序列 (Cutie 训练默认 seq_length=8)，每帧按 GOP 决定 I/P |
| Decoder 冻结 | 只冻 `final_conv`（1×1） | 冻整个 mask_decoder + transformer 套件 —— 隐式约束更强但灵活度低 |

---

## 十、推荐实施顺序

1. **环境准备**
   - 编译 `localAttention` CUDA op（github: zzd1992/Image-Local-Attention）
   - 用 AR-Seg 的 x265+libde265 工具链编码 DAVIS-2017 → 3 Mbps + GOP 12，提取 MV
2. **代码骨架**
   - 新建 `cutie/model/creff.py`, `cutie/model/mv_warp.py`, `cutie/inference/hr_pixel_memory.py`
   - 在 `cutie.py` 加 `encode_image_compressed`
   - 在 `train_wrapper.py` 加 GOP-aware forward
   - 在 `losses.py` 加 MSE 蒸馏项
3. **单步联调**
   - 加载官方 `cutie-base-mega.pth`，直接跑 inference（is_i=True 强制全 I）应当复现原 J&F
   - 把 GOP 改 8，开 CReFF，跑 inference，确认前向不报错；J&F 期望略降（CReFF 没训）
4. **Stage 1 训练**：DAVIS-2017 train + YouTube-VOS train，LR 0.5×，5e-5 AdamW，30 epoch
5. **Stage 2 训练**：开 CReFF 和 MSE 蒸馏，冻 decoder 全套，50 epoch
6. **评测**
   - DAVIS-2017 val：复现 AR-Seg 风格的精度-效率 Pareto
   - MOSEv1 → MOSEv2：核心战场
   - LVOS v2：长视频 d 退化曲线（GOP 12/30 对比）
7. **Ablation**（按 Cutie_AR-Seg迁移方案.md 7.3）

---

## 十一、关键文件路径速查（你日后写代码时）

| 任务 | 需要打开的文件 |
|---|---|
| 改主模型加 CReFF | `cutie/model/cutie.py`, `cutie/model/big_modules.py` |
| 加 CReFF/warp 模块 | `cutie/model/creff.py`（新建）, `cutie/model/mv_warp.py`（新建） |
| 改训练 forward | `cutie/model/train_wrapper.py`, `cutie/model/losses.py` |
| 改推理调度 | `cutie/inference/inference_core.py`, `cutie/inference/hr_pixel_memory.py`（新建） |
| 改数据集 | `cutie/dataset/vos_dataset.py` → 新版加 MV |
| 改配置 | `cutie/config/model/base.yaml` 加 `use_creff`, `creff_k`；`cutie/config/eval_config.yaml` 加 `gop_length`, `lr_scale` |

## 一句话总结

**AR-Seg 的 `forward_phase1/forward_phase2` 双阶段接口，等价于 Cutie 的 `encode_image` → `[CReFF 插入点]` → 余下所有模块**；只要在 `pix_feat` 一处做 MV warp + 局部 attention 融合，下游 memory、transformer、decoder 全部不动，就完成了从 VSS 到 VOS 的范式迁移，并自然继承 FST 的隐式共享 decoder 约束。
