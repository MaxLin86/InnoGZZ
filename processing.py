"""处理与控制模块：工作流协调、交互式界面、目录批处理、UI展示。"""

import argparse
import csv
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Tuple, Union

import cv2
import numpy as np

from algorithms import (
    DeblurProcessor,
    blur_laplacian_var,
    endoscopy_sharpness_score,
    imread_bgr,
    iter_video_samples,
    process_video_to_mp4,
    process_sample,
    select_best_frame_in_window,
)
from summary import (
    DeblurSelectionRecord,
    SampleMetrics,
    collect_media_files,
    detect_input_kind,
    ensure_dir,
    prefixed_name,
    print_summary,
    save_jpeg,
    write_deblur_selection_metadata,
    write_summary,
)


# ============================================================================
# 路径与交互式预览函数
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


def crop_left_third(image: np.ndarray) -> np.ndarray:
    """截断图像左侧三分之一的像素。"""
    width = image.shape[1]
    crop_start = width // 3
    return image[:, crop_start:].copy()


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
        print("Every-frame compression mode enabled (output as MP4)")
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
    last_deblur_frame = None
    saved_current_frame = None
    last_selected_frame_index = None
    last_selected_score = None
    last_selected_offset = None
    last_selected_blur_score = None
    last_deblur_blur_score = None
    last_select_elapsed_sec = None
    last_total_elapsed_sec = None
    last_visible_current_blur_score = None
    last_visible_current_sharpness_score = None

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

    def prefetch_temporal_window(center_index: int, current_frame: np.ndarray, current_timestamp: float) -> dict:
        """按顺序预读 temporal_unsharp 的候选窗口，避免每个候选帧都随机 seek。"""

        frame_cache = {center_index: (current_frame, current_timestamp)}
        if getattr(args, "deblur_mode", "unsharp") != "temporal_unsharp":
            return frame_cache

        radius = max(0, int(getattr(args, "temporal_radius", 6)))
        stride = max(1, int(getattr(args, "temporal_stride", 1)))
        first_index = max(0, center_index - radius)
        last_index = min(max(total_frames - 1, 0), center_index + radius)
        candidate_indices = set(range(first_index, last_index + 1, stride))

        capture.set(cv2.CAP_PROP_POS_FRAMES, first_index)
        for candidate_index in range(first_index, last_index + 1):
            ok, candidate_frame = capture.read()
            if not ok or candidate_frame is None:
                break
            if candidate_index in candidate_indices:
                frame_cache[candidate_index] = (candidate_frame, candidate_index / source_fps)

        return frame_cache

    def create_display_frame(frame: np.ndarray, deblur_frame: np.ndarray = None, 
                            current_timestamp: float = 0.0) -> np.ndarray:
        """创建整合显示帧：上方视频+操作区，下方预览面板。
        
        布局结构：
        - 上半部分 (1920×540):
          - 左侧: 视频窗口 960×540
          - 右侧: 操作文字区域 960×540 (黑色背景)
        - 下半部分 (1920×810):
          - 左: current 帧
          - 右: deblur 结果
        
        Args:
            frame: 当前视频帧
            deblur_frame: deblur 结果帧（可选）
            current_timestamp: 当前时间戳
        
        Returns:
            整合后的显示帧（1920×1350）
        """
        nonlocal last_visible_current_blur_score, last_visible_current_sharpness_score

        # 目标尺寸
        top_height = 540      # 上半部分高度
        bottom_height = 810   # 下半部分高度（预览面板）
        video_width = 960     # 视频窗口宽度
        info_width = 960      # 操作信息区宽度
        preview_width = 960   # 两张预览图各占一半宽度
        total_width = 1920    # 总宽度
        
        # === 上半部分：视频 + 操作信息 ===
        # 缩放视频帧到 960×540
        video_resized = cv2.resize(frame, (video_width, top_height), interpolation=cv2.INTER_AREA)
        
        # 创建黑色操作信息区
        info_panel = np.zeros((top_height, info_width, 3), dtype=np.uint8)
        
        # 添加操作说明文字（放大字体，三列布局）
        blur_score_current = blur_laplacian_var(frame)
        sharpness_score_current = endoscopy_sharpness_score(frame)
        last_visible_current_blur_score = blur_score_current
        last_visible_current_sharpness_score = sharpness_score_current
        
        # blur 对比只在 selected 与 deblur 之间进行，避免把 current 帧混入结果评估。
        if last_selected_blur_score is not None and last_deblur_blur_score is not None:
            deblur_blur_gain = last_deblur_blur_score - last_selected_blur_score
        else:
            deblur_blur_gain = None
        
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
            f"Current sharp: {sharpness_score_current:.2f}",
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
            "Saved Flow:",
            "",
            f"Selected frame: {last_selected_frame_index + 1}" if last_selected_frame_index is not None else "Selected frame: N/A",
            f"Selected offset: {last_selected_offset:+d}" if last_selected_offset is not None else "Selected offset: N/A",
            f"Selected sharp: {last_selected_score:.2f}" if last_selected_score is not None else "Selected sharp: N/A",
            f"Selected blur: {last_selected_blur_score:.2f}" if last_selected_blur_score is not None else "Selected blur: N/A",
            f"Deblur blur: {last_deblur_blur_score:.2f}" if last_deblur_blur_score is not None else "Deblur blur: N/A",
            f"Deblur vs selected: {deblur_blur_gain:+.2f}" if deblur_blur_gain is not None else "Deblur vs selected: N/A",
            f"Select time: {last_select_elapsed_sec * 1000.0:.1f}ms" if last_select_elapsed_sec is not None else "Select time: N/A",
            f"Total time: {last_total_elapsed_sec * 1000.0:.1f}ms" if last_total_elapsed_sec is not None else "Total time: N/A",
        ]
        
        for line in preview_lines:
            if line == "Saved Flow:":
                # 标题用白色加粗
                cv2.putText(info_panel, line, (col3_x, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(info_panel, line, (col3_x, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
            elif ":" in line:
                # 信息行统一绘制，分数和增益用更醒目的颜色。
                if "N/A" in line:
                    cv2.putText(info_panel, line, (col3_x, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(info_panel, line, (col3_x, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 2, cv2.LINE_AA)
                elif "gain:" in line.lower() or line.lower().startswith("deblur vs selected:"):
                    # 增益值根据正负显示不同颜色
                    if "+" in line or (deblur_blur_gain is not None and deblur_blur_gain > 0 and "-" not in line.split(":")[1].strip()):
                        cv2.putText(info_panel, line, (col3_x, y_offset), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
                        cv2.putText(info_panel, line, (col3_x, y_offset), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
                    else:
                        cv2.putText(info_panel, line, (col3_x, y_offset), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
                        cv2.putText(info_panel, line, (col3_x, y_offset), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
                elif "score:" in line.lower() or "blur:" in line.lower():
                    cv2.putText(info_panel, line, (col3_x, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(info_panel, line, (col3_x, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
                else:
                    cv2.putText(info_panel, line, (col3_x, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(info_panel, line, (col3_x, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2, cv2.LINE_AA)
            else:
                # 空行
                pass
            y_offset += 38
        
        # 水平拼接上半部分
        top_panel = cv2.hconcat([video_resized, info_panel])
        
        # === 下半部分：预览面板 ===
        if deblur_frame is not None and saved_current_frame is not None:
            # 显示 current 和 deblur；selected 只保存到文件，避免窗口过挤。
            cropped_current = crop_left_third(saved_current_frame)
            cropped_deblur = crop_left_third(deblur_frame)
            
            # 缩放到目标尺寸
            current_resized = cv2.resize(cropped_current, (preview_width, bottom_height), 
                                         interpolation=cv2.INTER_AREA)
            deblur_resized = cv2.resize(cropped_deblur, (preview_width, bottom_height), 
                                        interpolation=cv2.INTER_AREA)
            
            # 添加标签
            cv2.putText(current_resized, "Current", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(current_resized, "Current", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
            
            cv2.putText(deblur_resized, "Deblur", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(deblur_resized, "Deblur", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
            
            # 水平拼接
            bottom_panel = cv2.hconcat([current_resized, deblur_resized])
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
            display_frame = create_display_frame(frame, last_deblur_frame, timestamp_sec)
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
                frame_index = max(0, frame_index - 100)
                continue
            if key == ord("d"):
                frame_index = min(max(total_frames - 1, 0), frame_index + 100)
                continue
            if key == ord("s"):
                is_playing = False  # 按下S时停止自动播放
                total_start = time.perf_counter()
                
                # 保存 current 帧，用于下方预览面板显示。
                saved_current_frame = frame.copy()

                deblur_processor = DeblurProcessor(
                    mode=args.deblur_mode,
                    unsharp_amount=args.deblur_unsharp,
                    unsharp_sigma=getattr(args, "deblur_sigma", 1.2),
                )
                select_start = time.perf_counter()
                frame_cache = prefetch_temporal_window(frame_index, frame, timestamp_sec)

                def cached_read_frame(target_index: int) -> Tuple[np.ndarray, float]:
                    bounded_index = max(0, min(target_index, max(total_frames - 1, 0)))
                    cached = frame_cache.get(bounded_index)
                    if cached is not None:
                        return cached
                    return read_frame(bounded_index)

                (
                    selected_frame,
                    selected_frame_index,
                    selected_timestamp,
                    selected_score,
                    selected_offset,
                ) = select_best_frame_in_window(
                    center_index=frame_index,
                    frame_reader=cached_read_frame,
                    total_frames=total_frames,
                    video_fps=source_fps,
                    mode=getattr(args, "deblur_mode", "unsharp"),
                    temporal_radius=getattr(args, "temporal_radius", 6),
                    temporal_stride=getattr(args, "temporal_stride", 1),
                )
                select_elapsed_sec = time.perf_counter() - select_start

                # 文件名前缀包含保存序号、current 帧号和 selected 帧号。
                frame_sample_id = (
                    f"save_{len(saved_records):06d}"
                    f"_cur_f{frame_index + 1:06d}"
                    f"_sel_f{selected_frame_index + 1:06d}"
                    f"_t{timestamp_sec:08.3f}s"
                )
                current_path = sample_dir / prefixed_name(frame_sample_id, "current.jpg")
                selected_path = sample_dir / prefixed_name(frame_sample_id, "selected.jpg")
                deblur_path = sample_dir / prefixed_name(frame_sample_id, "deblur.jpg")
                metadata_path = sample_dir / prefixed_name(frame_sample_id, "deblur_metrics.json")

                deblur_frame = deblur_processor.apply(selected_frame)
                last_deblur_frame = deblur_frame  # 保存用于显示
                last_selected_frame_index = selected_frame_index
                last_selected_score = selected_score
                last_selected_offset = selected_offset
                current_sharpness_score = (
                    last_visible_current_sharpness_score
                    if last_visible_current_sharpness_score is not None
                    else endoscopy_sharpness_score(frame)
                )
                current_blur_score = (
                    last_visible_current_blur_score
                    if last_visible_current_blur_score is not None
                    else blur_laplacian_var(frame)
                )

                with ThreadPoolExecutor(max_workers=3) as executor:
                    current_future = executor.submit(save_jpeg, current_path, frame, args.frame_quality)
                    selected_future = executor.submit(save_jpeg, selected_path, selected_frame, args.frame_quality)
                    deblur_future = executor.submit(save_jpeg, deblur_path, deblur_frame, args.deblur_quality)
                    current_size = current_future.result()
                    selected_size = selected_future.result()
                    deblur_size = deblur_future.result()

                selected_blur_score = blur_laplacian_var(selected_frame)
                deblur_blur_score = blur_laplacian_var(deblur_frame)
                last_selected_blur_score = selected_blur_score
                last_deblur_blur_score = deblur_blur_score
                total_elapsed_sec = time.perf_counter() - total_start
                last_select_elapsed_sec = select_elapsed_sec
                last_total_elapsed_sec = total_elapsed_sec
                record = DeblurSelectionRecord(
                    sample_id=frame_sample_id,
                    source_path=str(input_path),
                    frame_index=frame_index,
                    timestamp_sec=timestamp_sec,
                    width=frame.shape[1],
                    height=frame.shape[0],
                    deblur_mode=args.deblur_mode,
                    current_jpg_bytes=current_size,
                    selected_jpg_bytes=selected_size,
                    deblur_jpg_bytes=deblur_size,
                    current_blur_score=current_blur_score,
                    selected_blur_score=selected_blur_score,
                    deblur_blur_score=deblur_blur_score,
                    deblur_vs_selected_blur_gain=deblur_blur_score - selected_blur_score,
                    current_sharpness_score=current_sharpness_score,
                    selected_frame_index=selected_frame_index,
                    selected_timestamp_sec=selected_timestamp,
                    selected_sharpness_score=selected_score,
                    selected_offset=selected_offset,
                    temporal_radius=getattr(args, "temporal_radius", None),
                    temporal_stride=getattr(args, "temporal_stride", None),
                    select_elapsed_sec=select_elapsed_sec,
                    total_elapsed_sec=total_elapsed_sec,
                )
                write_deblur_selection_metadata(metadata_path, record)
                saved_records.append(record)

                # 更新显示以显示预览面板
                display_frame = create_display_frame(frame, deblur_frame, timestamp_sec)
                cv2.imshow(window_name, display_frame)
                
                print(
                    "Saved frame: "
                    f"frame_index={frame_index}, "
                    f"time={timestamp_sec:.3f}s, "
                    f"selected_frame_index={selected_frame_index}, "
                    f"selected_offset={selected_offset:+d}, "
                    f"selected_score={selected_score:.2f}, "
                    f"current_blur_score={current_blur_score:.2f}, "
                    f"selected_blur_score={selected_blur_score:.2f}, "
                    f"deblur_blur_score={deblur_blur_score:.2f}, "
                    f"select_elapsed_ms={select_elapsed_sec * 1000.0:.1f}, "
                    f"total_elapsed_ms={total_elapsed_sec * 1000.0:.1f}, "
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
                
                total_selected_size = 0
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
                        total_selected_size += row["source_file_size"]
                    
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
                    if total_selected_size > 0:
                        avg_row["source_file_size"] = round(total_selected_size / video_count, 2)
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
