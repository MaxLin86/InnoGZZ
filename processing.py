"""处理与控制模块：工作流协调、交互式界面、目录批处理、UI展示。"""

import argparse
import csv
import json
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

from algorithms import (
    DeblurProcessor,
    blur_laplacian_var,
    imread_bgr,
    iter_video_samples,
    iter_video_all_frames,
    process_video_to_mp4,
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


# ============================================================================
# 交互式预览与显示函数
# ============================================================================


def output_dir_for_media(output_dir: Path, input_dir: Path, media_path: Path, every_frame_mode: bool = False) -> Path:
    """生成输出目录。
    
    Args:
        every_frame_mode: True=平铺模式（所有文件在同一目录），False=保持目录结构
    """
    if every_frame_mode:
        # 平铺模式：所有文件输出到同一个目录
        return output_dir
    else:
        # 正常模式：按输入目录层级生成输出目录
        relative_without_suffix = media_path.relative_to(input_dir).with_suffix("")
        return output_dir / relative_without_suffix


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


def crop_left_third(image: np.ndarray) -> np.ndarray:
    """截断图像左侧三分之一的像素。"""
    height, width = image.shape[:2]
    crop_start = width // 3
    return image[:, crop_start:].copy()


def make_combined_preview_panel(original_frame: np.ndarray, deblurred_frame: np.ndarray, 
                                target_width: int = 540, target_height: int = 540) -> np.ndarray:
    """创建整合预览面板：原图在右上，去模糊后的图在右下，并截断左侧三分之一像素。
    
    Args:
        original_frame: 原始帧
        deblurred_frame: 去模糊后的帧
        target_width: 目标宽度（默认540）
        target_height: 目标高度（默认540）
    
    Returns:
        竖直拼接的预览面板（原图在上，去模糊后在下）
    """
    
    # 先截断左侧三分之一像素
    cropped_original = crop_left_third(original_frame)
    cropped_deblurred = crop_left_third(deblurred_frame)
    
    # 缩放到目标大小
    original_resized = cv2.resize(cropped_original, (target_width, target_height), interpolation=cv2.INTER_AREA)
    deblurred_resized = cv2.resize(cropped_deblurred, (target_width, target_height), interpolation=cv2.INTER_AREA)
    
    # 竖直堆叠（上原图，下去模糊后）
    preview_panel = cv2.vconcat([original_resized, deblurred_resized])
    return preview_panel


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

    sample_dir = output_dir / input_path.stem
    sample_id = input_path.stem
    return process_image(input_path, sample_dir, args, sample_id)


def process_video_file(input_path: Path, output_dir: Path, args: argparse.Namespace) -> List[SampleMetrics]:
    """单视频压缩还原模式：按固定抽帧策略处理一个视频文件。"""

    sample_dir = output_dir / input_path.stem
    sample_id = input_path.stem
    return process_video(input_path, sample_dir, args, sample_id)


def process_video(
    input_path: Path,
    sample_dir: Path,
    args: argparse.Namespace,
    sample_base_id: str,
) -> Union[List[SampleMetrics], dict]:
    """视频入口：逐个处理抽出的帧，并收集所有样本指标.
    
    Returns:
        List[SampleMetrics] for normal mode, or dict metadata for every-frame mode
    """

    # 检查是否启用逐帧压缩模式（输出 MP4）
    every_frame_mode = getattr(args, "every_frame", False)
    
    if every_frame_mode:
        # 逐帧压缩模式：处理整个视频并输出 MP4 文件，不生成单帧结果
        print(f"Every-frame compression mode enabled (output as MP4)")
        _, _, metadata = process_video_to_mp4(
            input_path=input_path,
            output_dir=sample_dir,
            compression_scale=args.compression_scale,
            original_quality=args.original_quality,
            compressed_quality=args.compressed_quality,
            restored_quality=args.restored_quality,
            restore_sharpen=args.restore_sharpen,
            detail_enhance=args.detail_enhance,
            filename_prefix=sample_base_id.replace("/", "_"),
        )
        # 返回元数据字典
        return metadata
    
    # 正常模式：按 sample_fps 抽帧
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
    input_dir: Path,
    args: argparse.Namespace,
) -> List[DeblurSelectionRecord]:
    """视频模式：打开交互式界面，通过按键选帧并保存处理结果。
    
    Args:
        input_path: 输入视频路径
        output_dir: 总输出目录
        input_dir: 输入目录（用于计算相对路径）
        args: 参数对象
    """

    # 按输入目录层级生成输出子目录
    relative_without_suffix = input_path.relative_to(input_dir).with_suffix("")
    sample_dir = output_dir / relative_without_suffix
    ensure_dir(sample_dir)

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {input_path}")

    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if source_fps <= 0.0:
        source_fps = 30.0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_index = 0
    is_playing = True  # 默认开启自动播放
    saved_records: List[DeblurSelectionRecord] = []
    last_deblurred_frame = None  # 保存最后一次去模糊的帧
    saved_original_frame = None  # 保存第一次选中的原始帧（用于预览面板固定显示）

    window_name = "Video Frame Selector (1920x1350)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1920, 1350)

    print(f"Interactive video mode: {input_path}")
    print(f"Video info: total_frames={total_frames}, fps={source_fps:.3f}")
    print("Controls: a/d move ±100 frames, s save frame (stops auto-play), space play/pause, q skip video")

    def read_frame(target_index: int) -> Tuple[np.ndarray, float]:
        bounded_index = max(0, min(target_index, max(total_frames - 1, 0)))
        capture.set(cv2.CAP_PROP_POS_FRAMES, bounded_index)
        ok, frame = capture.read()
        if not ok or frame is None:
            raise ValueError(f"Failed to read frame {bounded_index} from: {input_path}")
        return frame, bounded_index / source_fps

    def create_display_frame(frame: np.ndarray, deblurred_frame: np.ndarray = None, 
                            current_timestamp: float = 0.0) -> np.ndarray:
        """创建整合显示帧：上方视频+操作区，下方预览面板。
        
        布局结构：
        - 上半部分 (1920×540):
          - 左侧: 视频窗口 960×540
          - 右侧: 操作文字区域 960×540 (黑色背景)
        - 下半部分 (1920×810):
          - 左下角: 原图 960×810
          - 右下角: 去模糊图 960×810
        
        Args:
            frame: 当前视频帧
            deblurred_frame: 去模糊后的帧（可选）
            current_timestamp: 当前时间戳
        
        Returns:
            整合后的显示帧（1920×1350）
        """
        # 目标尺寸
        top_height = 540      # 上半部分高度
        bottom_height = 810   # 下半部分高度（预览面板）
        video_width = 960     # 视频窗口宽度
        info_width = 960      # 操作信息区宽度
        preview_width = 960   # 每个预览图宽度
        total_width = 1920    # 总宽度
        total_height = 1350   # 总高度
        
        # === 上半部分：视频 + 操作信息 ===
        # 缩放视频帧到 960×540
        video_resized = cv2.resize(frame, (video_width, top_height), interpolation=cv2.INTER_AREA)
        
        # 创建黑色操作信息区
        info_panel = np.zeros((top_height, info_width, 3), dtype=np.uint8)
        
        # 添加操作说明文字（放大字体，三列布局）
        blur_score_current = blur_laplacian_var(frame)
        
        # 如果有保存的原始帧和去模糊帧，计算它们的模糊分数
        if saved_original_frame is not None and last_deblurred_frame is not None:
            blur_score_original = blur_laplacian_var(saved_original_frame)
            blur_score_deblurred = blur_laplacian_var(last_deblurred_frame)
            score_gain = blur_score_deblurred - blur_score_original
        else:
            blur_score_original = 0.0
            blur_score_deblurred = 0.0
            score_gain = 0.0
        
        # 三列布局：每列宽度约320像素
        col_width = info_width // 3  # 320像素
        col1_x = 20  # 第一列X坐标
        col2_x = col_width + 20  # 第二列X坐标
        col3_x = col_width * 2 + 20  # 第三列X坐标
        
        # === 第一列：Controls ===
        y_offset = 40
        controls_lines = [
            "Controls:",
            "",
            "[Space] Play/Pause",
            "[A] Back 100 frames",
            "[D] Forward 100 frames",
            "[S] Save current frame",
            "[Q] Skip this video",
        ]
        
        for line in controls_lines:
            if line == "Controls:":
                # 标题用白色加粗
                cv2.putText(info_panel, line, (col1_x, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(info_panel, line, (col1_x, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
            elif line.startswith("["):
                # 按键用绿色高亮
                cv2.putText(info_panel, line, (col1_x, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(info_panel, line, (col1_x, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                # 空行
                pass
            y_offset += 38
        
        # === 第二列：Status ===
        y_offset = 40
        status_lines = [
            "Status:",
            "",
            f"Frame: {frame_index + 1}/{total_frames}",
            f"Time: {current_timestamp:.3f}s",
            f"Current blur: {blur_score_current:.2f}",
        ]
        
        for line in status_lines:
            if line == "Status:":
                # 标题用白色加粗
                cv2.putText(info_panel, line, (col2_x, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(info_panel, line, (col2_x, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
            elif ":" in line and not line.startswith(" "):
                # 状态信息用灰色
                cv2.putText(info_panel, line, (col2_x, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(info_panel, line, (col2_x, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2, cv2.LINE_AA)
            else:
                # 空行
                pass
            y_offset += 38
        
        # === 第三列：Preview Panel ===
        y_offset = 40
        preview_lines = [
            "Preview Panel:",
            "",
            f"Original score: {blur_score_original:.2f}" if blur_score_original > 0 else "Original score: N/A",
            f"Deblurred score: {blur_score_deblurred:.2f}" if blur_score_deblurred > 0 else "Deblurred score: N/A",
            f"Score gain: {score_gain:+.2f}" if score_gain != 0 else "Score gain: N/A",
        ]
        
        for line in preview_lines:
            if line == "Preview Panel:":
                # 标题用白色加粗
                cv2.putText(info_panel, line, (col3_x, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(info_panel, line, (col3_x, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
            elif "score:" in line.lower() or "gain:" in line.lower():
                # 分数信息用黄色或绿色/红色
                if "N/A" in line:
                    cv2.putText(info_panel, line, (col3_x, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(info_panel, line, (col3_x, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 2, cv2.LINE_AA)
                elif "gain:" in line.lower():
                    # 增益值根据正负显示不同颜色
                    if "+" in line or (score_gain > 0 and "-" not in line.split(":")[1].strip()):
                        cv2.putText(info_panel, line, (col3_x, y_offset), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
                        cv2.putText(info_panel, line, (col3_x, y_offset), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
                    else:
                        cv2.putText(info_panel, line, (col3_x, y_offset), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
                        cv2.putText(info_panel, line, (col3_x, y_offset), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
                else:
                    cv2.putText(info_panel, line, (col3_x, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(info_panel, line, (col3_x, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            else:
                # 空行
                pass
            y_offset += 38
        
        # 水平拼接上半部分
        top_panel = cv2.hconcat([video_resized, info_panel])
        
        # === 下半部分：预览面板 ===
        if deblurred_frame is not None and saved_original_frame is not None:
            # 使用保存的原始帧（固定不变）和去模糊帧
            # 截断左侧三分之一并缩放到 960×810
            cropped_original = crop_left_third(saved_original_frame)
            cropped_deblurred = crop_left_third(deblurred_frame)
            
            # 缩放到目标尺寸
            original_resized = cv2.resize(cropped_original, (preview_width, bottom_height), 
                                         interpolation=cv2.INTER_AREA)
            deblurred_resized = cv2.resize(cropped_deblurred, (preview_width, bottom_height), 
                                          interpolation=cv2.INTER_AREA)
            
            # 添加标签
            cv2.putText(original_resized, "Original", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(original_resized, "Original", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
            
            cv2.putText(deblurred_resized, "Deblurred", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(deblurred_resized, "Deblurred", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
            
            # 水平拼接
            bottom_panel = cv2.hconcat([original_resized, deblurred_resized])
        else:
            # 没有去模糊帧时，创建黑色占位面板
            bottom_panel = np.zeros((bottom_height, total_width, 3), dtype=np.uint8)
            cv2.putText(bottom_panel, "Press 'S' to save and preview", (300, 650), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
        
        # === 垂直拼接上下两部分 ===
        display_frame = cv2.vconcat([top_panel, bottom_panel])
        
        return display_frame

    try:
        while True:
            frame, timestamp_sec = read_frame(frame_index)
            
            # 创建整合显示帧
            display_frame = create_display_frame(frame, last_deblurred_frame, timestamp_sec)
            cv2.imshow(window_name, display_frame)

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
                frame_index = max(0, frame_index - 100)
                continue
            if key == ord("d"):
                is_playing = False
                frame_index = min(max(total_frames - 1, 0), frame_index + 100)
                continue
            if key == ord("s"):
                is_playing = False  # 按下S时停止自动播放
                
                # 每次保存时都更新原始帧（用于预览面板显示）
                saved_original_frame = frame.copy()
                
                # 构建文件名（不包含路径前缀）
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
                last_deblurred_frame = deblurred_frame  # 保存用于显示
                
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
                    blur_score_original=blur_laplacian_var(frame),
                    blur_score_deblurred=blur_score_deblurred,
                    blur_score_gain=blur_score_deblurred - blur_laplacian_var(frame),
                )
                write_deblur_selection_metadata(metadata_path, record)
                saved_records.append(record)

                # 更新显示以显示预览面板
                display_frame = create_display_frame(frame, deblurred_frame, timestamp_sec)
                cv2.imshow(window_name, display_frame)
                
                print(
                    "Saved frame: "
                    f"frame_index={frame_index}, "
                    f"time={timestamp_sec:.3f}s, "
                    f"blur_score={blur_laplacian_var(frame):.2f}, "
                    f"deblurred_blur_score={blur_score_deblurred:.2f}, "
                    f"output_prefix={frame_sample_id}"
                )
                continue
    finally:
        capture.release()
        try:
            cv2.destroyWindow(window_name)
        except cv2.error:
            pass

    return saved_records


# ============================================================================
# 批处理与协调
# ============================================================================


def deblur_select_file(input_path: Path, output_dir: Path, args: argparse.Namespace) -> None:
    """单视频交互式去模糊选择入口。
    
    输出目录为：output_dir / input_path.stem
    """
    
    input_kind = detect_input_kind(input_path)
    if input_kind != "video":
        raise ValueError(f"deblur_select requires a video file, got: {input_path}")
    
    # 单视频模式：使用视频文件名（不含扩展名）作为输出子目录
    sample_dir = output_dir / input_path.stem
    ensure_dir(sample_dir)
    
    # 为了保持接口一致，传入 input_path.parent 作为 input_dir
    process_video_interactive(input_path, output_dir, input_path.parent, args)
    print(f"Interactive deblur selection complete. Results saved to: {sample_dir}")


def deblur_select_directory(input_dir: Path, output_dir: Path, args: argparse.Namespace) -> None:
    """目录批量交互式去模糊选择：遍历所有视频，依次打开交互式界面。
    
    保持输入目录的层级结构，在输出目录中创建对应的子目录。
    """
    
    # 仅搜集视频文件
    from summary import VIDEO_EXTENSIONS
    video_files = [
        p for p in sorted(input_dir.rglob("*"))
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    ]
    
    if not video_files:
        print(f"No video files found in: {input_dir}")
        return
    
    print(f"Found {len(video_files)} video(s) for interactive processing")
    print(f"Output mode: preserve directory structure under {output_dir}")
    
    for index, video_path in enumerate(video_files, start=1):
        print(f"\n[{index}/{len(video_files)}] Opening: {video_path}")
        
        try:
            process_video_interactive(video_path, output_dir, input_dir, args)
            print(f"✓ Completed: {video_path}")
        except Exception as exc:
            print(f"✗ Error processing {video_path}: {exc}")
            import traceback
            traceback.print_exc()
        
        print("-" * 80)
    
    print(f"\nAll videos processed. Results saved to: {output_dir}")


def process_input_directory(input_dir: Path, output_dir: Path, args: argparse.Namespace) -> List[SampleMetrics]:
    """目录入口：批量处理文件夹中的图片、视频或混合素材。"""

    media_files = collect_media_files(input_dir, output_dir)
    if not media_files:
        print(f"No supported image/video files found in: {input_dir}")
        return []

    # 检查是否启用逐帧压缩模式
    every_frame_mode = getattr(args, "every_frame", False)
    if every_frame_mode:
        print("Every-frame compression mode: skipping image test samples, processing only videos")
        # 过滤掉图像文件，只处理视频
        media_files = [(path, kind) for path, kind in media_files if kind == "video"]
        if not media_files:
            print(f"No video files found in: {input_dir}")
            return []

    all_metrics: List[SampleMetrics] = []
    all_video_metadata = []  # 存储视频元数据
    failures: List[Tuple[Path, str]] = []
    for index, (media_path, input_kind) in enumerate(media_files, start=1):
        relative_name = media_path.relative_to(input_dir).with_suffix("").as_posix()
        sample_dir = output_dir_for_media(output_dir, input_dir, media_path, every_frame_mode)
        print(f"[{index}/{len(media_files)}] Processing {input_kind}: {media_path}")

        try:
            if input_kind == "image":
                all_metrics.extend(process_image(media_path, sample_dir, args, relative_name))
            else:
                result = process_video(media_path, sample_dir, args, relative_name)
                if every_frame_mode and isinstance(result, dict):
                    # 在 every-frame 模式下，result 是元数据字典
                    all_video_metadata.append({
                        "video_path": str(media_path),
                        "metadata": result
                    })
                else:
                    # 在正常模式下，result 是 SampleMetrics 列表
                    all_metrics.extend(result)
        except Exception as exc:
            failures.append((media_path, str(exc)))
            print(f"Failed to process {media_path}: {exc}")

    # 保存合并的统计信息到单个 CSV 文件（仅 every-frame 模式）
    if every_frame_mode and all_video_metadata:
        # MP4 模式：保存视频元数据为 CSV
        stats_path = output_dir / "combined_statistics.csv"
        ensure_dir(output_dir)
        
        if all_video_metadata:
            # 获取所有字段名（从第一个视频的 metadata），并添加压缩率列
            fieldnames = list(all_video_metadata[0]["metadata"].keys()) + ["compression_ratio"]
            
            with stats_path.open("w", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                total_source_size = 0
                total_compressed_size = 0
                total_restored_size = 0
                total_processing_time = 0
                video_count = len(all_video_metadata)
                
                for item in all_video_metadata:
                    # 转换文件大小为 MB
                    row = item["metadata"].copy()
                    
                    source_mb = None
                    compressed_mb = None
                    restored_mb = None
                    
                    if "source_file_size" in row and row["source_file_size"] is not None:
                        source_mb = round(row["source_file_size"] / (1024 * 1024), 2)
                        row["source_file_size"] = source_mb
                        total_source_size += row["source_file_size"]
                    
                    if "compressed_file_size" in row and row["compressed_file_size"] is not None:
                        compressed_mb = round(row["compressed_file_size"] / (1024 * 1024), 2)
                        row["compressed_file_size"] = compressed_mb
                        total_compressed_size += row["compressed_file_size"]
                    
                    if "restored_file_size" in row and row["restored_file_size"] is not None:
                        restored_mb = round(row["restored_file_size"] / (1024 * 1024), 2)
                        row["restored_file_size"] = restored_mb
                        total_restored_size += row["restored_file_size"]
                    
                    # 计算压缩率（压缩后大小 / 原始大小 * 100%）
                    if source_mb and compressed_mb and source_mb > 0:
                        row["compression_ratio"] = round((compressed_mb / source_mb) * 100, 2)
                    else:
                        row["compression_ratio"] = None
                    
                    if "processing_time_sec" in row and row["processing_time_sec"] is not None:
                        total_processing_time += row["processing_time_sec"]
                    
                    writer.writerow(row)
                
                # 添加平均值行
                avg_row = {field: "" for field in fieldnames}
                avg_row[fieldnames[0]] = "AVERAGE"  # 在第一列标记为平均值
                
                if video_count > 0:
                    if total_source_size > 0:
                        avg_row["source_file_size"] = round(total_source_size / video_count, 2)
                    if total_compressed_size > 0:
                        avg_row["compressed_file_size"] = round(total_compressed_size / video_count, 2)
                    if total_restored_size > 0:
                        avg_row["restored_file_size"] = round(total_restored_size / video_count, 2)
                    if total_processing_time > 0:
                        avg_row["processing_time_sec"] = round(total_processing_time / video_count, 2)
                    
                    # 计算平均压缩率
                    if avg_row["source_file_size"] and avg_row["compressed_file_size"]:
                        avg_row["compression_ratio"] = round(
                            (avg_row["compressed_file_size"] / avg_row["source_file_size"]) * 100, 2
                        )
                
                writer.writerow(avg_row)
            
            print(f"Combined statistics saved to: {stats_path}")

    if failures:
        ensure_dir(output_dir)
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
        if input_path.is_file():
            deblur_select_file(input_path, output_dir, args)
        else:
            deblur_select_directory(input_path, output_dir, args)
    
    else:
        raise ValueError(f"Unknown task: {args.task}")
