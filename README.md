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
  - `summary_videos.csv`：仅展示视频输入的"单视频平均结果"，最后一行为全部视频的平均值。
  - `summary_video_frames.csv`：记录视频逐帧子结果，主要用于留档和排查。
  - `summary.json`：同时保存图片汇总、视频汇总和逐帧明细的结构化结果。
- 运动模糊去除调试集中在 `deblur_select`：`unsharp` 为单帧锐化基线，`temporal_unsharp` 会在当前帧前后搜索更清晰的邻近帧，再做亮度通道锐化。

## 代码结构

```
demo.py          # 命令行入口 (73行)
  ↓ imports
processing.py    # 工作流协调、交互式UI (390行)
  ↓ imports
algorithms.py    # 核心算法、图像处理 (350行)
  ↓ imports
summary.py       # 数据输出、表格生成 (520行)
```

### 分工：

| 模块 | 职责 |
|------|------|
| **demo.py** | 📋 CLI参数解析、命令行入口 |
| **algorithms.py** | 🔧 核心处理 `process_sample()`、压缩还原、质量评估、去模糊、视频读取 |
| **summary.py** | 📊 CSV/JSON表格、数据统计、元数据保存、文件操作 |
| **processing.py** | ⚙️ 工作流协调、交互式UI、目录批处理 |

### 关键函数分布：

**algorithms.py** 包含：
- 核心处理：`process_sample()` ← 统一的图像/视频帧处理函数
- 算法：`resize_by_scale()`, `restore_to_size()`, `psnr()`, `ssim_score()`, `blur_laplacian_var()`
- 去模糊：`DeblurProcessor`, `luminance_unsharp_mask()`, `endoscopy_sharpness_score()`
- 视频：`iter_video_samples()` (视频抽帧生成器)
- 图像IO：`imread_bgr()`, `save_jpeg_raw()`, `file_size()`

**processing.py** 包含：
- UI函数：`resize_for_preview()`, `overlay_preview_info()`, `make_preview_panel()`
- 交互式：`process_video_interactive()` (OpenCV窗口、按键控制)
- 样本处理入口：`process_image()`, `process_video()`, `process_image_file()`, `process_video_file()`
- 批处理：`process_input_directory()`, `run_batch()`

**summary.py** 包含：
- 数据类：`SampleMetrics`, `DeblurSelectionRecord`
- 文件操作：`ensure_dir()`, `collect_media_files()`, `detect_input_kind()`
- 表格生成：`build_metric_row()`, `aggregate_metric_rows()`, `build_summary_tables()`
- IO接口：`write_summary()`, `write_csv_rows()`, `save_jpeg()`

## 运行示例

处理单张图片：

```bash
python3 demo.py --task compress_restore --input /path/to/image_4k.jpg --output outputs/image_test
```

处理单个视频并进入交互式选帧：

```bash
python3 demo.py --task deblur_select --input /path/to/video_4k.mp4 --output outputs/video_ui_test --deblur-mode temporal_unsharp --temporal-radius 6
```

`deblur_select` 保存 `current.jpg`、`selected.jpg`、`deblur.jpg` 和 `deblur_metrics.json`：其中 `current.jpg` 是你按下 `S` 时的视频帧，`selected.jpg` 是 sharp score 胜出、实际用于处理的帧，`deblur.jpg` 是结果图。

处理一个输入文件夹：

```bash
python3 demo.py --task compress_restore --input /path/to/input_folder --output outputs/batch_test
```

文件夹内的视频默认每秒抽取 1 帧：

```bash
python3 demo.py --task compress_restore --input /path/to/input_folder --output outputs/batch_test --sample-fps 1
```

每个视频只调试前 5 个抽帧样本：

```bash
python3 demo.py --task compress_restore --input /path/to/input_folder --output outputs/batch_test --max-samples 5
```

**逐帧压缩模式**（新功能）：对视频的每一帧进行压缩及复原，跳过图像测试样本和原有的选帧逻辑：

```bash
python3 demo.py --task compress_restore --input /path/to/input_folder --output outputs/every_frame_test --every-frame-compression-scale 0.5 --original-quality 100 --compressed-quality 80 --restored-quality 100
```

此模式下：
- 处理视频的所有帧（不抽帧）
- 自动跳过文件夹中的图像文件
- 生成完整的统计表格（summary_images.csv、summary_videos.csv、summary_video_frames.csv、summary.json）

## 输出结构

```
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
