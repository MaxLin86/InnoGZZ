"""汇总与输出模块：CSV/JSON 表格生成、数据统计、元数据保存、文件操作。"""

import csv
import json
import mimetypes
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


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
    compression_scale: Optional[float] = None
    processing_time_sec: Optional[float] = None
    # MP4 模式专用字段
    source_file_size: Optional[int] = None
    compressed_file_size: Optional[int] = None
    restored_file_size: Optional[int] = None


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
    current_jpg_bytes: int
    selected_jpg_bytes: int
    deblur_jpg_bytes: int
    current_blur_score: float
    selected_blur_score: float
    deblur_blur_score: float
    deblur_vs_selected_blur_gain: float
    current_sharpness_score: Optional[float] = None
    selected_frame_index: Optional[int] = None
    selected_timestamp_sec: Optional[float] = None
    selected_sharpness_score: Optional[float] = None
    selected_offset: Optional[int] = None
    temporal_radius: Optional[int] = None
    temporal_stride: Optional[int] = None
    select_elapsed_sec: Optional[float] = None
    total_elapsed_sec: Optional[float] = None


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
    "compression_scale",
    "processing_time_sec",
    "source_file_size",
    "compressed_file_size",
    "restored_file_size",
]


def ensure_dir(path: Path) -> None:
    """确保输出目录存在，避免保存图片或报告时因为目录缺失失败。"""

    path.mkdir(parents=True, exist_ok=True)


def file_size(path: Path) -> int:
    """读取文件大小，单位为 bytes，用于后续比较 JPEG 压缩效果。"""

    return path.stat().st_size


def save_jpeg(path: Path, image_bgr: np.ndarray, quality: int) -> int:
    """按指定 JPEG 质量保存图像，并返回保存后的文件大小。"""

    from algorithms import save_jpeg_raw
    
    ensure_dir(path.parent)
    return save_jpeg_raw(path, image_bgr, quality)


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


def collect_media_files(input_dir: Path, output_dir: Path) -> list:
    """递归扫描输入目录，收集所有可识别的图片和视频文件。"""

    media_files = []
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


def prefixed_name(prefix: str, filename: str) -> str:
    """给视频帧输出文件加帧号前缀；图片输出则保持固定文件名。"""

    return f"{prefix}__{filename}" if prefix else filename


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
