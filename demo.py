#!/usr/bin/env python3
"""4K 图像/视频压缩、还原与运动模糊调试 Demo 入口。

代码拆分：
  - algorithms.py：压缩还原、质量指标、运动模糊去除算法。
  - io_control.py：输入目录扫描、输出保存、批处理流程和报告。
  - demo.py：只负责命令行参数和程序入口。
"""

import argparse
from pathlib import Path

from io_control import run_batch


def build_arg_parser() -> argparse.ArgumentParser:
    """定义命令行参数，便于独立调试压缩、还原和去模糊模块。"""

    parser = argparse.ArgumentParser(
        description="4K image/video compression, restoration, and deblur debug demo."
    )
    parser.add_argument("--input", required=True, help="Input directory containing images/videos.")
    parser.add_argument("--output", default="outputs", help="Output directory.")
    parser.add_argument(
        "--compression-scale",
        type=float,
        default=0.5,
        help="Resize scale for the compressed representation. 0.5 maps 4K UHD to 2K/FHD.",
    )
    parser.add_argument("--original-quality", type=int, default=95, help="JPEG quality for original_4k.jpg.")
    parser.add_argument("--compressed-quality", type=int, default=85, help="JPEG quality for compressed_2k.jpg.")
    parser.add_argument("--restored-quality", type=int, default=95, help="JPEG quality for restored_4k.jpg.")
    parser.add_argument("--restore-sharpen", type=float, default=0.35, help="Unsharp amount after restoration.")
    parser.add_argument(
        "--detail-enhance",
        action="store_true",
        help="Enable OpenCV detailEnhance after upsampling. Better visual edges, slower on 4K.",
    )
    parser.add_argument("--sample-fps", type=float, default=1.0, help="Video sample rate. Default: 1 frame/sec.")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit processed video samples.")
    parser.add_argument(
        "--deblur-mode",
        choices=["none", "unsharp", "wiener"],
        default="none",
        help="Debug interface for motion deblurring.",
    )
    parser.add_argument("--deblur-unsharp", type=float, default=0.55, help="Unsharp amount for deblur-mode=unsharp.")
    parser.add_argument("--motion-length", type=int, default=15, help="Motion kernel length for deblur-mode=wiener.")
    parser.add_argument("--motion-angle", type=float, default=0.0, help="Motion kernel angle for deblur-mode=wiener.")
    parser.add_argument("--wiener-noise", type=float, default=0.02, help="Noise power for deblur-mode=wiener.")
    return parser


def main() -> int:
    """程序入口：解析参数，并交给 io_control.run_batch() 执行。"""

    parser = build_arg_parser()
    args = parser.parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    run_batch(input_path, output_dir, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
