"""输入输出与流程控制模块：目录扫描、文件保存、批处理、单图模式和交互式视频模式。"""

import argparse
import csv
import json
import mimetypes
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

from algorithms import (
    DeblurProcessor,
    blur_laplacian_var,
    psnr,
    resize_by_scale,
    restore_to_size,
    ssim_score,
)


IMAGE_EXTENSIONS = {
    ".bmp",
    ".dib",
    ".jpeg",
    ".jpg",
    ".jpe",
    ".jp2",
    ".png",
    ".webp",
    ".pbm",
    ".pgm",
    ".ppm",
    ".pxm",
    ".pnm",
    ".tif",
    ".tiff",
}

VIDEO_EXTENSIONS = {
    ".avi",
    ".mp4",
    ".mov",
    ".mkv",
    ".m4v",
    ".webm",
    ".mpg",
    ".mpeg",
    ".ts",
}


@dataclass
class SampleMetrics:
    """单个样本的输出指标，最终会写入 metrics.json、summary.csv 和 summary.json。"""

    sample_id: str
    source_kind: str
    source_path: str
    frame_index: Optional[int]
    timestamp_sec: Optional[float]
    width: int
    height: int
    compressed_width: int
    compressed_height: int
    original_jpg_bytes: int
    compressed_2k_jpg_bytes: int
    restored_jpg_bytes: int
    jpg_size_ratio: float
    bytes_saved_percent: float
    psnr_restored: float
    ssim_restored: Optional[float]
    blur_laplacian_var_original: float
    blur_laplacian_var_restored: float
    deblur_mode: str
    deblurred_jpg_bytes: Optional[int] = None
    psnr_deblurred: Optional[float] = None
    ssim_deblurred: Optional[float] = None
    blur_laplacian_var_deblurred: Optional[float] = None


@dataclass
class DeblurSelectionRecord:
    """交互式选帧去模糊模式下的单帧记录。"""

    sample_id: str
    source_path: str
    frame_index: int
    timestamp_sec: float
    width: int
    height: int
    deblur_mode: str
    original_jpg_bytes: int
    deblurred_jpg_bytes: int
    blur_score_original: float
    blur_score_deblurred: float
    blur_score_gain: float


SUMMARY_FIELDS = [
    "row_type",
    "display_name",
    "group_name",
    "source_kind",
    "source_path",
    "sample_count",
    "frame_count",
    "width",
    "height",
    "compressed_width",
    "compressed_height",
    "original_jpg_bytes",
    "compressed_2k_jpg_bytes",
    "restored_jpg_bytes",
    "jpg_size_ratio",
    "bytes_saved_percent",
    "psnr_restored",
    "ssim_restored",
    "blur_laplacian_var_original",
    "blur_laplacian_var_restored",
    "deblur_mode",
    "deblurred_jpg_bytes",
    "psnr_deblurred",
    "ssim_deblurred",
    "blur_laplacian_var_deblurred",
]


def ensure_dir(path: Path) -> None:
    """确保输出目录存在，避免保存图片或报告时因为目录缺失失败。"""

    path.mkdir(parents=True, exist_ok=True)


def file_size(path: Path) -> int:
    """读取文件大小，单位为 bytes，用于后续比较 JPEG 压缩效果。"""

    return path.stat().st_size


def imread_bgr(path: Path) -> np.ndarray:
    """用 OpenCV 读取图像，并统一返回 BGR 格式。"""

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image


def save_jpeg(path: Path, image_bgr: np.ndarray, quality: int) -> int:
    """按指定 JPEG 质量保存图像，并返回保存后的文件大小。"""

    ensure_dir(path.parent)
    params = [
        int(cv2.IMWRITE_JPEG_QUALITY),
        int(quality),
        int(cv2.IMWRITE_JPEG_OPTIMIZE),
        1,
    ]
    ok = cv2.imwrite(str(path), image_bgr, params)
    if not ok:
        raise IOError(f"Failed to write JPEG: {path}")
    return file_size(path)


def detect_input_kind(path: Path) -> str:
    """根据文件扩展名和 MIME 类型判断输入是图像还是视频。"""

    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    if suffix in VIDEO_EXTENSIONS:
        return "video"

    mime, _ = mimetypes.guess_type(str(path))
    if mime:
        if mime.startswith("image/"):
            return "image"
        if mime.startswith("video/"):
            return "video"
    raise ValueError(f"Cannot infer input type from extension: {path}")


def is_relative_to(path: Path, parent: Path) -> bool:
    """兼容旧版 Python 的 Path.is_relative_to()，用于判断路径归属关系。"""

    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def collect_media_files(input_dir: Path, output_dir: Path) -> List[Tuple[Path, str]]:
    """递归扫描输入目录，收集所有可识别的图片和视频文件。"""

    media_files: List[Tuple[Path, str]] = []
    for path in sorted(input_dir.rglob("*")):
        if not path.is_file():
            continue
        if is_relative_to(path.resolve(), output_dir.resolve()):
            continue
        try:
            input_kind = detect_input_kind(path)
        except ValueError:
            continue
        media_files.append((path, input_kind))
    return media_files


def output_dir_for_media(output_dir: Path, input_dir: Path, media_path: Path) -> Path:
    """按输入目录层级生成输出目录，每个输入文件对应一个独立目录。"""

    relative_without_suffix = media_path.relative_to(input_dir).with_suffix("")
    return output_dir / relative_without_suffix


def output_dir_for_single_file(output_dir: Path, input_path: Path) -> Path:
    """为单图或单视频输入生成输出目录。"""

    return output_dir / input_path.stem


def prefixed_name(prefix: str, filename: str) -> str:
    """给视频帧输出文件加帧号前缀；图片输出则保持固定文件名。"""

    return f"{prefix}__{filename}" if prefix else filename


def process_sample(
    image_bgr: np.ndarray,
    source_kind: str,
    source_path: Path,
    sample_id: str,
    sample_dir: Path,
    args: argparse.Namespace,
    filename_prefix: str = "",
    frame_index: Optional[int] = None,
    timestamp_sec: Optional[float] = None,
) -> SampleMetrics:
    """处理单个图像样本，是图像输入和视频抽帧输入共用的核心函数。"""

    ensure_dir(sample_dir)

    # 1. 先把原始 4K 帧压缩到 2K，得到真正用于传输或存储的压缩表示。
    height, width = image_bgr.shape[:2]
    compressed_bgr = resize_by_scale(image_bgr, args.compression_scale)
    compressed_height, compressed_width = compressed_bgr.shape[:2]

    # 2. 准备输出路径：原图 JPEG、2K 压缩 JPEG、4K 还原 JPEG。
    original_path = sample_dir / prefixed_name(filename_prefix, "original_4k.jpg")
    compressed_path = sample_dir / prefixed_name(filename_prefix, "compressed_2k.jpg")
    restored_path = sample_dir / prefixed_name(filename_prefix, "restored_4k.jpg")
    
    # 3-1. 保存原图 JPEG、2K 压缩 JPEG
    original_size = save_jpeg(original_path, image_bgr, args.original_quality)
    compressed_size = save_jpeg(compressed_path, compressed_bgr, args.compressed_quality)
    
    # 3-2. 重新读取压缩图，并且还原到原图大小。
    reload_compressed_bgr = cv2.imread(compressed_path)
    restored_bgr = restore_to_size(
        reload_compressed_bgr,
        target_size=(width, height),
        sharpen_amount=args.restore_sharpen,
        detail_enhance=args.detail_enhance,
    )
    restored_size = save_jpeg(restored_path, restored_bgr, args.restored_quality)

    # 4. 压缩还原模式默认不做去模糊；只有显式启用时才额外输出 deblurred_4k.jpg。
    deblurred_size = None
    psnr_deblurred = None
    ssim_deblurred = None
    blur_deblurred = None
    deblur_mode = getattr(args, "deblur_mode", "none")
    if deblur_mode != "none":
        deblur_processor = DeblurProcessor(
            mode=deblur_mode,
            motion_length=args.motion_length,
            motion_angle=args.motion_angle,
            wiener_noise=args.wiener_noise,
            unsharp_amount=args.deblur_unsharp,
        )
        deblurred_bgr = deblur_processor.apply(restored_bgr)
        deblurred_path = sample_dir / prefixed_name(filename_prefix, "deblurred_4k.jpg")
        deblurred_size = save_jpeg(deblurred_path, deblurred_bgr, args.restored_quality)
        psnr_deblurred = psnr(image_bgr, deblurred_bgr)
        ssim_deblurred = ssim_score(image_bgr, deblurred_bgr)
        blur_deblurred = blur_laplacian_var(deblurred_bgr)

    # 5. 计算文件大小压缩比、节省比例和图像质量指标。
    ratio = original_size / compressed_size if compressed_size > 0 else float("inf")
    saved_percent = (1.0 - compressed_size / original_size) * 100.0 if original_size > 0 else 0.0

    metrics = SampleMetrics(
        sample_id=sample_id,
        source_kind=source_kind,
        source_path=str(source_path),
        frame_index=frame_index,
        timestamp_sec=timestamp_sec,
        width=width,
        height=height,
        compressed_width=compressed_width,
        compressed_height=compressed_height,
        original_jpg_bytes=original_size,
        compressed_2k_jpg_bytes=compressed_size,
        restored_jpg_bytes=restored_size,
        jpg_size_ratio=ratio,
        bytes_saved_percent=saved_percent,
        psnr_restored=psnr(image_bgr, restored_bgr),
        ssim_restored=ssim_score(image_bgr, restored_bgr),
        blur_laplacian_var_original=blur_laplacian_var(image_bgr),
        blur_laplacian_var_restored=blur_laplacian_var(restored_bgr),
        deblur_mode=deblur_mode,
        deblurred_jpg_bytes=deblurred_size,
        psnr_deblurred=psnr_deblurred,
        ssim_deblurred=ssim_deblurred,
        blur_laplacian_var_deblurred=blur_deblurred,
    )

    write_sample_metadata(sample_dir / prefixed_name(filename_prefix, "metrics.json"), metrics)
    return metrics


def write_sample_metadata(path: Path, metrics: SampleMetrics) -> None:
    """把单个样本的指标写到当前样本目录下的 metrics.json。"""

    data = asdict(metrics)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def write_deblur_selection_metadata(path: Path, record: DeblurSelectionRecord) -> None:
    """把交互式去模糊单帧记录写到当前样本目录下的 JSON 文件。"""

    with path.open("w", encoding="utf-8") as handle:
        json.dump(asdict(record), handle, indent=2, ensure_ascii=False)


def average_optional(values: List[Optional[float]]) -> Optional[float]:
    """计算可选浮点值的平均值；空值会被自动跳过。"""

    valid_values = [value for value in values if value is not None]
    if not valid_values:
        return None
    return sum(valid_values) / len(valid_values)


def metric_group_name(metric: SampleMetrics) -> str:
    """为视频帧提取所属视频名；图片则直接返回自身样本名。"""

    if metric.source_kind == "video" and "/" in metric.sample_id:
        return metric.sample_id.rsplit("/", 1)[0]
    return metric.sample_id


def metric_display_name(metric: SampleMetrics) -> str:
    """生成表格展示名称；视频帧只展示最后一级帧名。"""

    if metric.source_kind == "video" and "/" in metric.sample_id:
        return metric.sample_id.rsplit("/", 1)[-1]
    return metric.sample_id


def build_metric_row(
    metric: SampleMetrics,
    row_type: str,
    display_name: str,
    group_name: str = "",
) -> Dict[str, Any]:
    """把单个样本指标转换成可写入 CSV/JSON 的表格行。"""

    return {
        "row_type": row_type,
        "display_name": display_name,
        "group_name": group_name,
        "source_kind": metric.source_kind,
        "source_path": metric.source_path,
        "sample_count": 1,
        "frame_count": 1 if metric.source_kind == "video" else 0,
        "width": metric.width,
        "height": metric.height,
        "compressed_width": metric.compressed_width,
        "compressed_height": metric.compressed_height,
        "original_jpg_bytes": metric.original_jpg_bytes,
        "compressed_2k_jpg_bytes": metric.compressed_2k_jpg_bytes,
        "restored_jpg_bytes": metric.restored_jpg_bytes,
        "jpg_size_ratio": metric.jpg_size_ratio,
        "bytes_saved_percent": metric.bytes_saved_percent,
        "psnr_restored": metric.psnr_restored,
        "ssim_restored": metric.ssim_restored,
        "blur_laplacian_var_original": metric.blur_laplacian_var_original,
        "blur_laplacian_var_restored": metric.blur_laplacian_var_restored,
        "deblur_mode": metric.deblur_mode,
        "deblurred_jpg_bytes": metric.deblurred_jpg_bytes,
        "psnr_deblurred": metric.psnr_deblurred,
        "ssim_deblurred": metric.ssim_deblurred,
        "blur_laplacian_var_deblurred": metric.blur_laplacian_var_deblurred,
    }


def aggregate_metric_rows(
    metrics: List[SampleMetrics],
    row_type: str,
    display_name: str,
    source_kind: str,
    source_path: str,
    group_name: str = "",
) -> Dict[str, Any]:
    """对一组样本做平均汇总，用于图片总平均和单视频总平均。"""

    if not metrics:
        raise ValueError("Cannot aggregate empty metrics.")

    return {
        "row_type": row_type,
        "display_name": display_name,
        "group_name": group_name,
        "source_kind": source_kind,
        "source_path": source_path,
        "sample_count": len(metrics),
        "frame_count": len(metrics) if source_kind == "video" else 0,
        "width": average_optional([float(item.width) for item in metrics]),
        "height": average_optional([float(item.height) for item in metrics]),
        "compressed_width": average_optional([float(item.compressed_width) for item in metrics]),
        "compressed_height": average_optional([float(item.compressed_height) for item in metrics]),
        "original_jpg_bytes": average_optional([float(item.original_jpg_bytes) for item in metrics]),
        "compressed_2k_jpg_bytes": average_optional([float(item.compressed_2k_jpg_bytes) for item in metrics]),
        "restored_jpg_bytes": average_optional([float(item.restored_jpg_bytes) for item in metrics]),
        "jpg_size_ratio": average_optional([item.jpg_size_ratio for item in metrics]),
        "bytes_saved_percent": average_optional([item.bytes_saved_percent for item in metrics]),
        "psnr_restored": average_optional([item.psnr_restored for item in metrics]),
        "ssim_restored": average_optional([item.ssim_restored for item in metrics]),
        "blur_laplacian_var_original": average_optional([item.blur_laplacian_var_original for item in metrics]),
        "blur_laplacian_var_restored": average_optional([item.blur_laplacian_var_restored for item in metrics]),
        "deblur_mode": metrics[0].deblur_mode,
        "deblurred_jpg_bytes": average_optional([item.deblurred_jpg_bytes for item in metrics]),
        "psnr_deblurred": average_optional([item.psnr_deblurred for item in metrics]),
        "ssim_deblurred": average_optional([item.ssim_deblurred for item in metrics]),
        "blur_laplacian_var_deblurred": average_optional([item.blur_laplacian_var_deblurred for item in metrics]),
    }


def aggregate_table_rows(
    rows: List[Dict[str, Any]],
    row_type: str,
    display_name: str,
    source_kind: str,
) -> Dict[str, Any]:
    """对已经生成的表格行再做一次平均，适合视频主表的最终平均行。"""

    if not rows:
        raise ValueError("Cannot aggregate empty rows.")

    def collect(key: str) -> List[Optional[float]]:
        return [row[key] for row in rows]

    return {
        "row_type": row_type,
        "display_name": display_name,
        "group_name": "",
        "source_kind": source_kind,
        "source_path": "",
        "sample_count": len(rows),
        "frame_count": sum(int(row["frame_count"]) for row in rows),
        "width": average_optional(collect("width")),
        "height": average_optional(collect("height")),
        "compressed_width": average_optional(collect("compressed_width")),
        "compressed_height": average_optional(collect("compressed_height")),
        "original_jpg_bytes": average_optional(collect("original_jpg_bytes")),
        "compressed_2k_jpg_bytes": average_optional(collect("compressed_2k_jpg_bytes")),
        "restored_jpg_bytes": average_optional(collect("restored_jpg_bytes")),
        "jpg_size_ratio": average_optional(collect("jpg_size_ratio")),
        "bytes_saved_percent": average_optional(collect("bytes_saved_percent")),
        "psnr_restored": average_optional(collect("psnr_restored")),
        "ssim_restored": average_optional(collect("ssim_restored")),
        "blur_laplacian_var_original": average_optional(collect("blur_laplacian_var_original")),
        "blur_laplacian_var_restored": average_optional(collect("blur_laplacian_var_restored")),
        "deblur_mode": rows[0]["deblur_mode"],
        "deblurred_jpg_bytes": average_optional(collect("deblurred_jpg_bytes")),
        "psnr_deblurred": average_optional(collect("psnr_deblurred")),
        "ssim_deblurred": average_optional(collect("ssim_deblurred")),
        "blur_laplacian_var_deblurred": average_optional(collect("blur_laplacian_var_deblurred")),
    }


def build_summary_tables(metrics: List[SampleMetrics]) -> Dict[str, Any]:
    """构建图片汇总、视频汇总和视频逐帧明细三类统计结果。"""

    image_metrics = [item for item in metrics if item.source_kind == "image"]
    video_metrics = [item for item in metrics if item.source_kind == "video"]

    image_rows = [
        build_metric_row(item, row_type="image", display_name=metric_display_name(item))
        for item in image_metrics
    ]
    image_average = (
        aggregate_table_rows(
            image_rows,
            row_type="image_average",
            display_name="AVERAGE",
            source_kind="image",
        )
        if image_rows
        else None
    )

    video_groups: Dict[str, List[SampleMetrics]] = {}
    for item in video_metrics:
        video_groups.setdefault(metric_group_name(item), []).append(item)

    video_rows: List[Dict[str, Any]] = []
    video_frame_rows: List[Dict[str, Any]] = []
    video_json_items: List[Dict[str, Any]] = []
    for group_name, group_items in video_groups.items():
        group_items = sorted(group_items, key=lambda item: (item.frame_index is None, item.frame_index, item.timestamp_sec))
        source_path = group_items[0].source_path
        video_summary = aggregate_metric_rows(
            group_items,
            row_type="video_summary",
            display_name=group_name,
            source_kind="video",
            source_path=source_path,
        )
        frame_rows = [
            build_metric_row(
                item,
                row_type="video_frame",
                display_name=f"  {metric_display_name(item)}",
                group_name=group_name,
            )
            for item in group_items
        ]
        video_rows.append(video_summary)
        video_frame_rows.extend(frame_rows)
        video_json_items.append({"summary": video_summary, "frames": frame_rows})

    video_average = (
        aggregate_table_rows(
            video_rows,
            row_type="video_average",
            display_name="AVERAGE",
            source_kind="video",
        )
        if video_rows
        else None
    )

    return {
        "images": {
            "rows": image_rows,
            "average": image_average,
            "count": len(image_rows),
        },
        "videos": {
            "rows": video_rows,
            "average": video_average,
            "count": len(video_rows),
            "frame_count": len(video_frame_rows),
            "details": video_frame_rows,
            "items": video_json_items,
        },
    }


def write_csv_rows(path: Path, rows: List[Dict[str, Any]]) -> None:
    """把表格行写入 CSV 文件。"""

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_summary(output_dir: Path, metrics: List[SampleMetrics]) -> Dict[str, Any]:
    """把所有样本按图片/视频拆分汇总，并写入 JSON/CSV。"""

    ensure_dir(output_dir)
    summary = build_summary_tables(metrics)
    json_path = output_dir / "summary.json"
    image_csv_path = output_dir / "summary_images.csv"
    video_csv_path = output_dir / "summary_videos.csv"
    video_frame_csv_path = output_dir / "summary_video_frames.csv"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    image_rows = list(summary["images"]["rows"])
    if summary["images"]["average"] is not None:
        image_rows.append(summary["images"]["average"])
    write_csv_rows(image_csv_path, image_rows)

    video_rows = list(summary["videos"]["rows"])
    if summary["videos"]["average"] is not None:
        video_rows.append(summary["videos"]["average"])
    write_csv_rows(video_csv_path, video_rows)
    write_csv_rows(video_frame_csv_path, summary["videos"]["details"])
    return summary


def process_image(
    input_path: Path,
    sample_dir: Path,
    args: argparse.Namespace,
    sample_id: str,
) -> List[SampleMetrics]:
    """图像入口：读取单张图像，并调用 process_sample() 完成主流程。"""

    image = imread_bgr(input_path)
    return [
        process_sample(
            image,
            source_kind="image",
            source_path=input_path,
            sample_id=sample_id,
            sample_dir=sample_dir,
            args=args,
        )
    ]


def process_image_file(input_path: Path, output_dir: Path, args: argparse.Namespace) -> List[SampleMetrics]:
    """单图模式：直接处理一张输入图像。"""

    sample_dir = output_dir_for_single_file(output_dir, input_path)
    sample_id = input_path.stem
    return process_image(input_path, sample_dir, args, sample_id)


def process_video_file(input_path: Path, output_dir: Path, args: argparse.Namespace) -> List[SampleMetrics]:
    """单视频压缩还原模式：按固定抽帧策略处理一个视频文件。"""

    sample_dir = output_dir_for_single_file(output_dir, input_path)
    sample_id = input_path.stem
    return process_video(input_path, sample_dir, args, sample_id)


def iter_video_samples(
    input_path: Path,
    sample_fps: float,
    max_samples: Optional[int],
) -> Iterable[Tuple[str, np.ndarray, int, float]]:
    """视频抽帧生成器：默认按 1 FPS 输出帧、帧序号和时间戳。"""

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {input_path}")

    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if source_fps <= 0.0:
        source_fps = 30.0

    # 将“每秒抽几帧”转换为“每隔多少原始帧取一次”。
    interval = max(1, int(round(source_fps / sample_fps)))
    frame_index = 0
    yielded = 0

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        if frame_index % interval == 0:
            timestamp_sec = frame_index / source_fps
            sample_id = f"frame_{yielded:06d}_t{timestamp_sec:08.3f}s"
            yield sample_id, frame, frame_index, timestamp_sec
            yielded += 1
            if max_samples is not None and yielded >= max_samples:
                break

        frame_index += 1
        if frame_count and frame_index >= frame_count:
            break

    capture.release()


def process_video(
    input_path: Path,
    sample_dir: Path,
    args: argparse.Namespace,
    sample_base_id: str,
) -> List[SampleMetrics]:
    """视频入口：逐个处理抽出的帧，并收集所有样本指标。"""

    metrics: List[SampleMetrics] = []
    for frame_sample_id, frame, frame_index, timestamp_sec in iter_video_samples(
        input_path,
        sample_fps=args.sample_fps,
        max_samples=args.max_samples,
    ):
        sample_id = f"{sample_base_id}/{frame_sample_id}"
        metrics.append(
            process_sample(
                frame,
                source_kind="video",
                source_path=input_path,
                sample_id=sample_id,
                sample_dir=sample_dir,
                args=args,
                filename_prefix=frame_sample_id,
                frame_index=frame_index,
                timestamp_sec=timestamp_sec,
            )
        )
    return metrics


def resize_for_preview(frame_bgr: np.ndarray, preview_scale: float) -> np.ndarray:
    """把预览帧缩放到适合交互窗口显示的尺寸。"""

    if preview_scale <= 0:
        preview_scale = 1.0
    if abs(preview_scale - 1.0) < 1e-6:
        return frame_bgr
    height, width = frame_bgr.shape[:2]
    preview_width = max(1, int(round(width * preview_scale)))
    preview_height = max(1, int(round(height * preview_scale)))
    return cv2.resize(frame_bgr, (preview_width, preview_height), interpolation=cv2.INTER_AREA)


def overlay_preview_info(
    frame_bgr: np.ndarray,
    frame_index: int,
    total_frames: int,
    timestamp_sec: float,
    blur_score: float,
    step: int,
    mode_text: str,
) -> np.ndarray:
    """在预览帧上叠加调试信息和按键提示。"""

    preview = frame_bgr.copy()
    lines = [
        f"frame: {frame_index + 1}/{total_frames}",
        f"time: {timestamp_sec:.3f}s",
        f"blur_score: {blur_score:.2f}",
        f"step: {step}",
        f"mode: {mode_text}",
        "keys: a/d move  j/l jump  -/+ step  space play/pause",
        "keys: s save current frame  q quit",
    ]
    start_y = 28
    for line in lines:
        cv2.putText(preview, line, (16, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(preview, line, (16, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 1, cv2.LINE_AA)
        start_y += 28
    return preview


def make_preview_panel(images: List[np.ndarray], max_height: int = 420) -> np.ndarray:
    """把多张图拼成横向调试面板，便于观察去模糊前后差异。"""

    panels = []
    for image in images:
        height, width = image.shape[:2]
        scale = min(1.0, max_height / float(height))
        target_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
        resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
        panels.append(resized)
    return cv2.hconcat(panels)


def process_video_interactive(
    input_path: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> List[DeblurSelectionRecord]:
    """视频模式：打开交互式界面，通过按键选帧并保存处理结果。"""

    sample_dir = output_dir_for_single_file(output_dir, input_path)
    ensure_dir(sample_dir)

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {input_path}")

    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if source_fps <= 0.0:
        source_fps = 30.0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_index = 0
    step = max(1, int(getattr(args, "video_seek_step", 1)))
    is_playing = False
    saved_records: List[DeblurSelectionRecord] = []

    window_name = "Video Frame Selector"
    preview_name = "Processed Preview"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.namedWindow(preview_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1440, 900)
    cv2.resizeWindow(preview_name, 1440, 520)

    print(f"Interactive video mode: {input_path}")
    print(f"Video info: total_frames={total_frames}, fps={source_fps:.3f}, default_step={step}")
    print("Controls: a/d move, j/l jump, -/+ change step, space play/pause, s save frame, q quit")

    def read_frame(target_index: int) -> Tuple[np.ndarray, float]:
        bounded_index = max(0, min(target_index, max(total_frames - 1, 0)))
        capture.set(cv2.CAP_PROP_POS_FRAMES, bounded_index)
        ok, frame = capture.read()
        if not ok or frame is None:
            raise ValueError(f"Failed to read frame {bounded_index} from: {input_path}")
        return frame, bounded_index / source_fps

    try:
        while True:
            frame, timestamp_sec = read_frame(frame_index)
            blur_score = blur_laplacian_var(frame)
            preview_frame = resize_for_preview(frame, getattr(args, "preview_scale", 0.5))
            overlay = overlay_preview_info(
                preview_frame,
                frame_index=frame_index,
                total_frames=total_frames,
                timestamp_sec=timestamp_sec,
                blur_score=blur_score,
                step=step,
                mode_text="play" if is_playing else "pause",
            )
            cv2.imshow(window_name, overlay)

            wait_ms = max(1, int(round(1000.0 / source_fps))) if is_playing else 0
            key = cv2.waitKey(wait_ms) & 0xFF
            if key == 255 and is_playing:
                if frame_index < max(total_frames - 1, 0):
                    frame_index += 1
                continue

            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                is_playing = not is_playing
                continue
            if key == ord("a"):
                is_playing = False
                frame_index = max(0, frame_index - step)
                continue
            if key == ord("d"):
                is_playing = False
                frame_index = min(max(total_frames - 1, 0), frame_index + step)
                continue
            if key == ord("j"):
                is_playing = False
                frame_index = max(0, frame_index - step * 10)
                continue
            if key == ord("l"):
                is_playing = False
                frame_index = min(max(total_frames - 1, 0), frame_index + step * 10)
                continue
            if key in (ord("-"), ord("_")):
                step = max(1, step // 2)
                continue
            if key in (ord("="), ord("+")):
                step = min(max(total_frames, 1), step * 2)
                continue
            if key == ord("s"):
                is_playing = False
                frame_sample_id = f"frame_{len(saved_records):06d}_t{timestamp_sec:08.3f}s"
                original_path = sample_dir / prefixed_name(frame_sample_id, "selected.jpg")
                deblurred_path = sample_dir / prefixed_name(frame_sample_id, "deblurred.jpg")
                metadata_path = sample_dir / prefixed_name(frame_sample_id, "deblur_metrics.json")

                deblur_processor = DeblurProcessor(
                    mode=args.deblur_mode,
                    motion_length=args.motion_length,
                    motion_angle=args.motion_angle,
                    wiener_noise=args.wiener_noise,
                    unsharp_amount=args.deblur_unsharp,
                )
                deblurred_frame = deblur_processor.apply(frame)
                original_size = save_jpeg(original_path, frame, args.selected_quality)
                deblurred_size = save_jpeg(deblurred_path, deblurred_frame, args.deblurred_quality)

                blur_score_deblurred = blur_laplacian_var(deblurred_frame)
                record = DeblurSelectionRecord(
                    sample_id=frame_sample_id,
                    source_path=str(input_path),
                    frame_index=frame_index,
                    timestamp_sec=timestamp_sec,
                    width=frame.shape[1],
                    height=frame.shape[0],
                    deblur_mode=args.deblur_mode,
                    original_jpg_bytes=original_size,
                    deblurred_jpg_bytes=deblurred_size,
                    blur_score_original=blur_score,
                    blur_score_deblurred=blur_score_deblurred,
                    blur_score_gain=blur_score_deblurred - blur_score,
                )
                write_deblur_selection_metadata(metadata_path, record)
                saved_records.append(record)

                cv2.imshow(preview_name, make_preview_panel([frame, deblurred_frame]))
                print(
                    "Saved frame: "
                    f"frame_index={frame_index}, "
                    f"time={timestamp_sec:.3f}s, "
                    f"blur_score={blur_score:.2f}, "
                    f"deblurred_blur_score={blur_score_deblurred:.2f}, "
                    f"output_prefix={frame_sample_id}"
                )
                continue
    finally:
        capture.release()
        try:
            cv2.destroyWindow(window_name)
            cv2.destroyWindow(preview_name)
        except cv2.error:
            pass

    return saved_records


def process_input_directory(input_dir: Path, output_dir: Path, args: argparse.Namespace) -> List[SampleMetrics]:
    """目录入口：批量处理文件夹中的图片、视频或混合素材。"""

    media_files = collect_media_files(input_dir, output_dir)
    if not media_files:
        print(f"No supported image/video files found in: {input_dir}")
        return []

    all_metrics: List[SampleMetrics] = []
    failures: List[Tuple[Path, str]] = []
    for index, (media_path, input_kind) in enumerate(media_files, start=1):
        relative_name = media_path.relative_to(input_dir).with_suffix("").as_posix()
        sample_dir = output_dir_for_media(output_dir, input_dir, media_path)
        print(f"[{index}/{len(media_files)}] Processing {input_kind}: {media_path}")

        try:
            if input_kind == "image":
                all_metrics.extend(process_image(media_path, sample_dir, args, relative_name))
            else:
                all_metrics.extend(process_video(media_path, sample_dir, args, relative_name))
        except Exception as exc:
            failures.append((media_path, str(exc)))
            print(f"Failed to process {media_path}: {exc}")

    if failures:
        failures_path = output_dir / "failures.json"
        with failures_path.open("w", encoding="utf-8") as handle:
            json.dump(
                [{"path": str(path), "error": error} for path, error in failures],
                handle,
                indent=2,
                ensure_ascii=False,
            )
        print(f"Failed files: {len(failures)}. See {failures_path}")

    return all_metrics


def print_summary(summary: Dict[str, Any], output_dir: Path) -> None:
    """在终端打印图片和视频分开展示的关键汇总信息。"""

    image_count = summary["images"]["count"]
    video_count = summary["videos"]["count"]
    video_frame_count = summary["videos"]["frame_count"]
    if image_count == 0 and video_count == 0:
        print("No samples were processed.")
        return

    print(f"Output directory: {output_dir}")
    print(f"Image inputs: {image_count}")
    if summary["images"]["average"] is not None:
        image_avg = summary["images"]["average"]
        image_message = (
            "Image average: "
            f"ratio={image_avg['jpg_size_ratio']:.2f}x, "
            f"saved={image_avg['bytes_saved_percent']:.2f}%, "
            f"PSNR={image_avg['psnr_restored']:.2f} dB"
        )
        if image_avg["ssim_restored"] is not None:
            image_message += f", SSIM={image_avg['ssim_restored']:.4f}"
        else:
            image_message += ", SSIM unavailable"
        print(image_message)
    print(f"Video inputs: {video_count}")
    print(f"Video frame records: {video_frame_count}")
    if summary["videos"]["average"] is not None:
        video_avg = summary["videos"]["average"]
        video_message = (
            "Video average: "
            f"ratio={video_avg['jpg_size_ratio']:.2f}x, "
            f"saved={video_avg['bytes_saved_percent']:.2f}%, "
            f"PSNR={video_avg['psnr_restored']:.2f} dB"
        )
        if video_avg["ssim_restored"] is not None:
            video_message += f", SSIM={video_avg['ssim_restored']:.4f}"
        else:
            video_message += ", SSIM unavailable"
        print(video_message)
    print(f"Image summary CSV: {output_dir / 'summary_images.csv'}")
    print(f"Video summary CSV: {output_dir / 'summary_videos.csv'}")
    print(f"Video frame detail CSV: {output_dir / 'summary_video_frames.csv'}")
    print(f"Summary JSON: {output_dir / 'summary.json'}")


def write_deblur_selection_summary(output_dir: Path, records: List[DeblurSelectionRecord]) -> None:
    """把交互式选帧去模糊结果写入单独的 CSV 和 JSON。"""

    ensure_dir(output_dir)
    json_path = output_dir / "deblur_selection.json"
    csv_path = output_dir / "deblur_selection.csv"
    fieldnames = [
        "sample_id",
        "source_path",
        "frame_index",
        "timestamp_sec",
        "width",
        "height",
        "deblur_mode",
        "original_jpg_bytes",
        "deblurred_jpg_bytes",
        "blur_score_original",
        "blur_score_deblurred",
        "blur_score_gain",
    ]

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump([asdict(record) for record in records], handle, indent=2, ensure_ascii=False)

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def print_deblur_selection_summary(records: List[DeblurSelectionRecord], output_dir: Path) -> None:
    """在终端打印交互式选帧去模糊结果摘要。"""

    print(f"Output directory: {output_dir}")
    print(f"Saved deblur frames: {len(records)}")
    if records:
        avg_gain = sum(record.blur_score_gain for record in records) / len(records)
        print(f"Average blur-score gain: {avg_gain:.2f}")
    print(f"Deblur selection CSV: {output_dir / 'deblur_selection.csv'}")
    print(f"Deblur selection JSON: {output_dir / 'deblur_selection.json'}")


def run_batch(input_path: Path, output_dir: Path, args: argparse.Namespace) -> List[Any]:
    """校验输入路径，并按压缩还原或选帧去模糊两种任务执行。"""

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    ensure_dir(output_dir)
    if args.task == "compress_restore":
        if input_path.is_dir():
            metrics = process_input_directory(input_path, output_dir, args)
        else:
            input_kind = detect_input_kind(input_path)
            if input_kind == "image":
                metrics = process_image_file(input_path, output_dir, args)
            else:
                metrics = process_video_file(input_path, output_dir, args)
        summary = write_summary(output_dir, metrics)
        print_summary(summary, output_dir)
        return metrics

    if args.task == "deblur_select":
        if input_path.is_dir():
            raise ValueError("deblur_select mode only supports a single video file input.")
        if detect_input_kind(input_path) != "video":
            raise ValueError("deblur_select mode requires a video file input.")
        records = process_video_interactive(input_path, output_dir, args)
        write_deblur_selection_summary(output_dir, records)
        print_deblur_selection_summary(records, output_dir)
        return records

    raise ValueError(f"Unsupported task: {args.task}")
