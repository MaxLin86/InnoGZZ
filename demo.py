#!/usr/bin/env python3
"""4K 图像/视频压缩、还原与运动模糊调试 Demo 入口。

代码拆分：
  - algorithms.py：压缩还原、质量评估、去模糊算法、核心处理流程。
  - summary.py：CSV/JSON 表格生成、数据统计、元数据保存、文件操作。
  - processing.py：工作流协调、交互式界面、目录批处理、UI展示。
  - demo.py：只负责命令行参数和程序入口。
"""

import argparse
import sys
from pathlib import Path

from processing import run_batch


def build_parser(task: str) -> argparse.ArgumentParser:
    """根据任务类型构建参数解析器。"""
    parser = argparse.ArgumentParser(description="4K image/video compression, restoration, and deblur debug demo.")
    parser.add_argument("--task", required=True, choices=["compress_restore", "deblur_select"], help="Choose exactly one function: compression/restoration or frame-selection/deblur.")
    parser.add_argument("--input", required=True, help="Input path for the selected task.")
    parser.add_argument("--output", default="outputs", help="Output directory.")

    if task == "compress_restore":
        parser.add_argument("--compression-scale", type=float, default=0.5, help="Resize scale for the compressed representation. 0.5 maps 4K UHD to 2K/FHD.")
        parser.add_argument("--original-quality", type=int, default=95, help="JPEG quality for original_4k.jpg.")
        parser.add_argument("--compressed-quality", type=int, default=85, help="JPEG quality for compressed_2k.jpg.")
        parser.add_argument("--restored-quality", type=int, default=95, help="JPEG quality for restored_4k.jpg.")
        parser.add_argument("--restore-sharpen", type=float, default=0.35, help="Unsharp amount after restoration.")
        parser.add_argument("--detail-enhance", action="store_true", help="Enable OpenCV detailEnhance after upsampling. Better visual edges, slower on 4K.")
        parser.add_argument("--sample-fps", type=float, default=1.0, help="Video sample rate. Default: 1 frame/sec.")
        parser.add_argument("--max-samples", type=int, default=None, help="Limit processed video samples.")
        parser.add_argument("--every-frame", action="store_true", help="Enable per-frame compression for videos. When set, compress and restore every frame and output as MP4 files (compressed.mp4 and restored.mp4), skipping image test samples and frame selection logic.")
    elif task == "deblur_select":
        parser.add_argument("--deblur-mode", choices=["unsharp"], default="unsharp", help="Deblur algorithm used in interactive selection mode (only unsharp supported).")
        parser.add_argument("--deblur-unsharp", type=float, default=0.55, help="Unsharp amount for deblur-mode=unsharp.")
        parser.add_argument("--selected-quality", type=int, default=95, help="JPEG quality for saved selected frames.")
        parser.add_argument("--deblurred-quality", type=int, default=95, help="JPEG quality for saved deblurred frames.")
    else:
        raise ValueError(f"Unsupported task: {task}")

    return parser


def parse_args() -> argparse.Namespace:
    """解析命令行参数（两阶段：先获取 task，再加载对应参数）。"""
    # 第一阶段：快速提取 task
    temp = argparse.ArgumentParser(add_help=False)
    temp.add_argument("--task", required=True, choices=["compress_restore", "deblur_select"])
    known, _ = temp.parse_known_args()
    
    # 第二阶段：根据 task 构建完整解析器
    parser = build_parser(known.task)
    return parser.parse_args()


def main() -> int:
    """程序入口：解析参数，并交给 io_control.run_batch() 执行。"""

    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    run_batch(input_path, output_dir, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
