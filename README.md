# 4K 图像/视频压缩还原 Demo

这个工作区提供一个可直接调试的 `demo.py`，核心流程是：

- 输入既可以是单张图片、单个视频，也可以是一个包含图片和视频的文件夹。
- 运行时必须明确指定 `--task`，并且同一次运行只能启动一个功能。
- `compress_restore`：只做压缩与还原。
- `deblur_select`：只做视频选帧与去模糊。
- 单张图片会直接进入单帧处理模式。
- 单个视频会进入交互式选帧界面，通过按键挑选要保存和处理的帧。
- 输入文件夹时会走批处理模式，自动扫描其中的图片和视频。
- 输出目录保持输入文件夹的目录层级；每张图片或每个视频都会有自己的输出目录。
- 图片输出 `original_4k.jpg`、`compressed_2k.jpg`、`restored_4k.jpg`。
- 视频的所有抽帧结果保存在同一个视频目录中，通过文件名中的帧号区分。
- 对比原图保存为 JPEG 的大小与缩放到 2K 后保存为 JPEG 的大小。
- 输出分层统计文件：
- `summary_images.csv`：仅展示图像输入结果，最后一行为平均值。
- `summary_videos.csv`：仅展示视频输入的“单视频平均结果”，最后一行为全部视频的平均值。
- `summary_video_frames.csv`：记录视频逐帧子结果，主要用于留档和排查。
- `summary.json`：同时保存图片汇总、视频汇总和逐帧明细的结构化结果。
- 运动模糊去除已提供 `none/unsharp/wiener` 调试接口，后续可以替换为深度学习模型。

## 代码结构

```text
demo.py        # 命令行入口，只负责解析参数和启动流程
algorithms.py  # 算法模块：压缩还原、指标计算、去模糊接口
io_control.py  # 输入输出控制：单图、单视频交互、目录批处理、文件保存、报告输出
```

## 运行示例

处理单张图片：

```bash
python3 demo.py --task compress_restore --input /path/to/image_4k.jpg --output outputs/image_test --deblur-mode none
```

处理单个视频并进入交互式选帧：

```bash
python3 demo.py --task deblur_select --input /path/to/video_4k.mp4 --output outputs/video_ui_test --preview-scale 0.5 --video-seek-step 1 --deblur-mode unsharp
```

处理一个输入文件夹：

```bash
python3 demo.py --task compress_restore --input /path/to/input_folder --output outputs/batch_test --deblur-mode none
```

文件夹内的视频默认每秒抽取 1 帧：

```bash
python3 demo.py --task compress_restore --input /path/to/input_folder --output outputs/batch_test --sample-fps 1 --deblur-mode none
```

每个视频只调试前 5 个抽帧样本：

```bash
python3 demo.py --task compress_restore --input /path/to/input_folder --output outputs/batch_test --max-samples 5 --deblur-mode none
```

## 输出结构

```text
outputs/
  summary_images.csv
  summary_videos.csv
  summary_video_frames.csv
  summary.json
  scene_a/
    image_a/
      original_4k.jpg
      compressed_2k.jpg
      restored_4k.jpg
      metrics.json
    video_a/
      frame_000000_t0000.000s__original_4k.jpg
      frame_000000_t0000.000s__compressed_2k.jpg
      frame_000000_t0000.000s__restored_4k.jpg
      frame_000000_t0000.000s__metrics.json
      frame_000001_t0001.000s__original_4k.jpg
      frame_000001_t0001.000s__compressed_2k.jpg
      frame_000001_t0001.000s__restored_4k.jpg
      frame_000001_t0001.000s__metrics.json
```

其中：

- `summary_images.csv` 只看图片输入。
- `summary_videos.csv` 只看每个视频的平均结果，逐帧明细默认不混在主表里。
- `summary_video_frames.csv` 保留逐帧记录，便于后续排查具体帧。

如果启用去模糊，还会额外输出：

```text
deblurred_4k.jpg
frame_000000_t0000.000s__deblurred_4k.jpg
```

交互式去模糊模式还会额外输出：

```text
deblur_selection.csv
deblur_selection.json
frame_000000_t0000.000s__selected.jpg
frame_000000_t0000.000s__deblurred.jpg
frame_000000_t0000.000s__deblur_metrics.json
```

## 视频交互模式按键

- `a / d`：按当前步长前后移动
- `j / l`：快速大步前后跳
- `- / +`：减小或增大步长
- `space`：播放 / 暂停
- `s`：保存当前帧，并执行压缩、还原、去模糊
- `q`：退出交互界面

## 算法说明

当前压缩还原链路采用传统图像算法，便于离线调试：

1. 使用 `cv2.INTER_AREA` 将 4K 图像按 `--compression-scale 0.5` 缩放到 2K/FHD。
2. 使用 JPEG 保存压缩结果，默认质量为 `85`。
3. 使用 Lanczos 插值还原回原始尺寸。
4. 使用轻量 unsharp mask 增强边缘，默认强度 `0.35`。
5. 用 PSNR/SSIM 评估还原图和原图的相似度。

如果后续要接入深度学习超分模型，可以保留 `process_sample()` 的输入输出协议，只替换 `restore_to_size()` 内部实现。

## 运动模糊接口

`DeblurProcessor.apply()` 是后续接入运动去模糊算法的位置。当前支持：

- `none`：默认，不做去模糊。
- `unsharp`：快速锐化基线，适合先验证后处理链路。
- `wiener`：传统运动核 Wiener 去卷积，参数为 `--motion-length`、`--motion-angle`、`--wiener-noise`。

后续可在该类中替换为深度学习模型，例如 MPRNet、Restormer、NAFNet 或自研模型推理接口。
