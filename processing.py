"""处理与控制模块：工作流协调、交互式界面、目录批处理、UI展示。"""

import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from algorithms import (
    DeblurProcessor,
    blur_laplacian_var,
    imread_bgr,
    iter_video_samples,
    psnr,
    process_sample,
    ssim_score,
)
from summary import (
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
    DeblurSelectionRecord,
    SampleMetrics,
    collect_media_files,
    detect_input_kind,
    ensure_dir,
    is_relative_to,
    metric_display_name,
    metric_group_name,
    prefixed_name,
    print_summary,
    save_jpeg,
    write_deblur_selection_metadata,
    write_sample_metadata,
    write_summary,
)


def output_dir_for_media(output_dir: Path, input_dir: Path, media_path: Path) -> Path:
    """按输入目录层级生成输出目录，每个输入文件对应一个独立目录。"""

    relative_without_suffix = media_path.relative_to(input_dir).with_suffix("")
    return output_dir / relative_without_suffix


def output_dir_for_single_file(output_dir: Path, input_path: Path) -> Path:
    """为单图或单视频输入生成输出目录。"""

    return output_dir / input_path.stem


# ============================================================================
# 交互式预览与显示函数
# ============================================================================


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


def make_preview_panel(images: list, max_height: int = 420) -> np.ndarray:
    """把多张图拼成横向调试面板，便于观察去模糊前后差异。"""

    panels = []
    for image in images:
        height, width = image.shape[:2]
        scale = min(1.0, max_height / float(height))
        target_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
        resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
        panels.append(resized)
    return cv2.hconcat(panels)


# ============================================================================
# 样本处理入口
# ============================================================================


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
            compression_scale=args.compression_scale,
            original_quality=args.original_quality,
            compressed_quality=args.compressed_quality,
            restored_quality=args.restored_quality,
            restore_sharpen=args.restore_sharpen,
            detail_enhance=args.detail_enhance,
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
                compression_scale=args.compression_scale,
                original_quality=args.original_quality,
                compressed_quality=args.compressed_quality,
                restored_quality=args.restored_quality,
                restore_sharpen=args.restore_sharpen,
                detail_enhance=args.detail_enhance,
                deblur_mode=getattr(args, "deblur_mode", "none"),
                motion_length=getattr(args, "motion_length", 15),
                motion_angle=getattr(args, "motion_angle", 0.0),
                wiener_noise=getattr(args, "wiener_noise", 0.02),
                deblur_unsharp=getattr(args, "deblur_unsharp", 0.55),
                filename_prefix=frame_sample_id,
                frame_index=frame_index,
                timestamp_sec=timestamp_sec,
            )
        )
    return metrics


# ============================================================================
# 交互式操作
# ============================================================================


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
            preview_frame = overlay_preview_info(
                frame,
                frame_index=frame_index,
                total_frames=total_frames,
                timestamp_sec=timestamp_sec,
                blur_score=blur_score,
                step=step,
                mode_text="play" if is_playing else "pause",
            )
            cv2.imshow(window_name, preview_frame)

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


# ============================================================================
# 批处理与协调
# ============================================================================


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


def run_batch(input_path: Path, output_dir: Path, args: argparse.Namespace) -> None:
    """总体协调函数：根据任务类型分发到压缩还原或交互式选帧模式。"""

    if args.task == "compress_restore":
        if input_path.is_file():
            input_kind = detect_input_kind(input_path)
            if input_kind == "image":
                metrics = process_image_file(input_path, output_dir, args)
            else:
                metrics = process_video_file(input_path, output_dir, args)
        else:
            metrics = process_input_directory(input_path, output_dir, args)
        
        if metrics:
            summary = write_summary(output_dir, metrics)
            print_summary(summary, output_dir)
    
    elif args.task == "deblur_select":
        if not input_path.is_file():
            raise ValueError("deblur_select requires a single video file as input, not a directory.")
        input_kind = detect_input_kind(input_path)
        if input_kind != "video":
            raise ValueError("deblur_select requires a video file as input.")
        
        process_video_interactive(input_path, output_dir, args)
        print(f"Interactive deblur selection complete. Results saved to: {output_dir}")
    
    else:
        raise ValueError(f"Unknown task: {args.task}")
