#  preprocess
## 1. ffmpeg

```bash
sudo apt update
sudo apt install ffmpeg
```

安装好后 `ffmpeg -version` 应该能看到版本信息。

---

## 2. x265（编码器）
https://github.com/videolan/x265/tree/419182243fb2e2dfbe91dfc45a51778cf704f849

下载后放脚本目录下的 `./x265`，你需要自己 build 一次：

```bash
cd ./x265/build
cmake ../source
make -j$(nproc)
cd ../..
```

编译完成后会得到一个可执行文件 `./x265/build/x265`，脚本会在工作目录里自动建立软链接。

---

## 3. libde265（解码器）

https://github.com/AlbertHuyb/libde265-MV/tree/53b7b04cd9f7c6c65c288d1335b6a2ef4a51c2a7

下载后同样需要编译：

```bash
cd ./libde265
mkdir build
cd build
cmake ..
make -j$(nproc)
cd ../..
```

编译完成后会在 `./libde265/build/dec265/dec265` 下生成解码器可执行文件 `dec265`。

---

## 4. 验证安装

* 确认 `ffmpeg` 在系统路径：

  ```bash
  which ffmpeg
  ```
* 确认 `x265` 可执行文件存在：

  ```bash
  ./x265/build/x265 --help
  ```
* 确认 `dec265` 可执行文件存在：

  ```bash
  ./libde265/build/dec265/dec265 --help
  ```

## 5. 先生成 MOT17 COCO 标注

`generate_mot17_mv.py` 需要 `train_half.json` 和 `val_half.json`，原因是脚本不是按帧号硬编码切分 train/val，而是读取 COCO JSON 里的 `images[*].file_name`，构建每个序列对应的帧集合。

这样做有两个目的：

* 保证预处理输出与 ByteTrack 训练、验证使用的半段划分完全一致；
* 只在 `(ref, cur)` 两帧都属于同一个 split 时才生成样本，避免 GOP / MV 跨过 train 和 val 边界造成数据泄漏。

因此第一次处理 MOT17 前，需要先在项目根目录生成 annotations：

```bash
cd /root/autodl-tmp/ByteTrack-ARSeg
python tools/convert_mot17_to_coco.py
```

生成后应当能看到：

```text
datasets/mot/annotations/train_half.json
datasets/mot/annotations/val_half.json
datasets/mot/annotations/train.json
datasets/mot/annotations/test.json
```

如果你的 MOT17 放在 `/root/autodl-tmp/ByteTrack/datasets/mot`，就在对应项目根目录执行同一个转换脚本，或者把下面命令里的路径改成实际位置。

---

## 6. 数据处理

建议在 `preprocess/` 目录执行，因为脚本会优先使用当前工作目录下的 `./x265/build/x265` 和 `./libde265/build/dec265/dec265`。

标准 COCO JSON 只处理 MOT17 的 FRCNN 序列，因此默认使用 `--detector_only FRCNN`：

```bash
cd /root/autodl-tmp/ByteTrack-ARSeg/preprocess

nohup /root/miniconda3/envs/bytetrack/bin/python generate_mot17_mv.py \
  --mot_root /root/autodl-tmp/ByteTrack-ARSeg/datasets/mot \
  --out_root /root/autodl-tmp/ByteTrack-ARSeg/datasets/compressed_mot \
  --ref_gap 12 --fps 30 --bitrate 3000 \
  --train_json /root/autodl-tmp/ByteTrack-ARSeg/datasets/mot/annotations/train_half.json \
  --val_json   /root/autodl-tmp/ByteTrack-ARSeg/datasets/mot/annotations/val_half.json \
  --detector_only FRCNN --jobs 3 \
  > mot17_preprocess.log 2>&1 &
```
旧命令
cd /root/autodl-tmp/ByteTrack-ARSeg/preprocess
nohup python generate_mot17_mv.py \
  --mot_root /root/autodl-tmp/ByteTrack-ARSeg/datasets/mot \
  --out_root /root/autodl-tmp/ByteTrack-ARSeg/datasets/compressed_mot \
  --ref_gap 12 --fps 30 --bitrate 3000 \
  --train_json /root/autodl-tmp/ByteTrack-ARSeg/datasets/mot/annotations/train_half.json \
  --val_json   /root/autodl-tmp/ByteTrack-ARSeg/datasets/mot/annotations/val_half.json \
  --detector_only FRCNN --jobs 3 \
  > mot17_preprocess.log 2>&1 &
```

---
