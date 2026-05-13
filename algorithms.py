"""算法模块：压缩还原、图像质量评估、运动模糊去除接口、图像读写与处理。"""

import math
import time
from pathlib import Path
from typing import Iterable, Optional, Tuple

import cv2
import numpy as np

from summary import SampleMetrics, ensure_dir, file_size, prefixed_name, save_jpeg, write_sample_metadata

try:
    from skimage.metrics import structural_similarity
except Exception:  # pragma: no cover - 可选依赖缺失时跳过 SSIM
    structural_similarity = None


def resize_by_scale(image_bgr: np.ndarray, scale: float) -> np.ndarray:
    """按比例缩放图像；默认 0.5 可将 4K UHD 压到接近 2K/FHD 尺寸。"""

    if not 0.0 < scale <= 1.0:
        raise ValueError("--compression-scale must be in (0, 1].")
    height, width = image_bgr.shape[:2]
    target_width = max(1, int(round(width * scale)))
    target_height = max(1, int(round(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(image_bgr, (target_width, target_height), interpolation=interpolation)


def restore_to_size(
    compressed_bgr: np.ndarray,
    target_size: Tuple[int, int],
    sharpen_amount: float,
    detail_enhance: bool,
) -> np.ndarray:
    """将压缩图像还原到目标尺寸，并进行可选的细节增强和锐化。"""

    restored = cv2.resize(compressed_bgr, target_size, interpolation=cv2.INTER_LANCZOS4)
    if detail_enhance:
        # OpenCV 的边缘保持增强适合肉眼调试，但处理 4K 帧会更慢。
        restored = cv2.detailEnhance(restored, sigma_s=4, sigma_r=0.08)
    if sharpen_amount > 0:
        restored = unsharp_mask(restored, amount=sharpen_amount, sigma=1.0)
    return restored


def unsharp_mask(image_bgr: np.ndarray, amount: float = 0.35, sigma: float = 1.0) -> np.ndarray:
    """非锐化掩膜：用原图减去高斯模糊图，提升边缘清晰度。"""

    blurred = cv2.GaussianBlur(image_bgr, (0, 0), sigmaX=sigma, sigmaY=sigma)
    sharpened = cv2.addWeighted(image_bgr, 1.0 + amount, blurred, -amount, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def psnr(original_bgr: np.ndarray, candidate_bgr: np.ndarray) -> float:
    """计算 PSNR，数值越高表示候选图越接近原图。"""

    mse = np.mean((original_bgr.astype(np.float32) - candidate_bgr.astype(np.float32)) ** 2)
    if mse <= 1e-12:
        return 100.0  # 两图完全相同，MSE ≈ 0 时返回固定值而非 inf
    return 20.0 * math.log10(255.0 / math.sqrt(float(mse)))


def ssim_score(original_bgr: np.ndarray, candidate_bgr: np.ndarray) -> Optional[float]:
    """计算灰度图 SSIM，数值越接近 1 表示结构相似度越高。"""

    if structural_similarity is None:
        return None
    original_gray = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)
    candidate_gray = cv2.cvtColor(candidate_bgr, cv2.COLOR_BGR2GRAY)
    return float(structural_similarity(original_gray, candidate_gray, data_range=255))


def blur_laplacian_var(image_bgr: np.ndarray) -> float:
    """使用拉普拉斯方差估计清晰度，数值越低通常表示越模糊。"""

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


class DeblurProcessor:
    """运动模糊去除的统一调试接口。

    模式：
      - none：不处理，直接返回输入图。
      - unsharp：快速锐化基线，适合先验证后处理链路。

    后续如需接入深度学习模型，可以只替换 apply() 内部逻辑。
    """

    def __init__(
        self,
        mode: str,
        unsharp_amount: float,
    ) -> None:
        """保存去模糊参数，便于 CLI 调试不同模式。
        
        Args:
            mode: 去模糊模式（none/unsharp）
            unsharp_amount: Unsharp Mask 锐化强度
        """

        self.mode = mode
        self.unsharp_amount = unsharp_amount

    def apply(self, image_bgr: np.ndarray) -> np.ndarray:
        """根据 deblur_mode 执行对应去模糊策略。"""

        if self.mode == "none":
            return image_bgr
        if self.mode == "unsharp":
            return unsharp_mask(image_bgr, amount=self.unsharp_amount, sigma=1.2)
        raise ValueError(f"Unsupported deblur mode: {self.mode}")


# ============================================================================
# 图像读写与操作函数
# ============================================================================


def file_size(path: Path) -> int:
    """读取文件大小，单位为 bytes，用于后续比较 JPEG 压缩效果。"""

    return path.stat().st_size


def imread_bgr(path: Path) -> np.ndarray:
    """用 OpenCV 读取图像，并统一返回 BGR 格式。"""

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image


def save_jpeg_raw(path: Path, image_bgr: np.ndarray, quality: int) -> int:
    """按指定 JPEG 质量保存图像（原始版本），并返回保存后的文件大小。
    
    注意：此函数假设目录已存在。用于内部使用。
    """

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

    # 将"每秒抽几帧"转换为"每隔多少原始帧取一次"。
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


def iter_video_all_frames(
    input_path: Path,
) -> Iterable[Tuple[str, np.ndarray, int, float]]:
    """视频逐帧生成器：遍历视频的每一帧，不进行抽帧。"""

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {input_path}")

    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if source_fps <= 0.0:
        source_fps = 30.0

    frame_index = 0

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        timestamp_sec = frame_index / source_fps
        sample_id = f"frame_{frame_index:06d}_t{timestamp_sec:08.3f}s"
        yield sample_id, frame, frame_index, timestamp_sec
        frame_index += 1

    capture.release()


# ============================================================================
# 核心处理函数
# ============================================================================


def process_sample(
    image_bgr: np.ndarray,
    source_kind: str,
    source_path: Path,
    sample_id: str,
    sample_dir: Path,
    compression_scale: float,
    original_quality: int,
    compressed_quality: int,
    restored_quality: int,
    restore_sharpen: float,
    detail_enhance: bool,
    deblur_mode: str = "none",
    deblur_unsharp: float = 0.55,
    filename_prefix: str = "",
    frame_index: Optional[int] = None,
    timestamp_sec: Optional[float] = None,
) -> "SampleMetrics":
    """处理单个图像样本，是图像输入和视频抽帧输入共用的核心函数。
    
    将原始图像进行压缩、还原，计算质量指标，并可选地进行去模糊处理。
    """
    
    start_time = time.time()
    ensure_dir(sample_dir)

    # 1. 先把原始 4K 帧压缩到 2K，得到真正用于传输或存储的压缩表示。
    height, width = image_bgr.shape[:2]
    compressed_bgr = resize_by_scale(image_bgr, compression_scale)
    compressed_height, compressed_width = compressed_bgr.shape[:2]

    # 2. 准备输出路径：原图 JPEG、2K 压缩 JPEG、4K 还原 JPEG。
    original_path = sample_dir / prefixed_name(filename_prefix, "original_4k.jpg")
    compressed_path = sample_dir / prefixed_name(filename_prefix, "compressed_2k.jpg")
    restored_path = sample_dir / prefixed_name(filename_prefix, "restored_4k.jpg")
    
    # 3-1. 保存原图 JPEG、2K 压缩 JPEG
    original_size = save_jpeg(original_path, image_bgr, original_quality)
    compressed_size = save_jpeg(compressed_path, compressed_bgr, compressed_quality)
    
    # 3-2. 重新读取压缩图，并且还原到原图大小。
    reload_compressed_bgr = cv2.imread(str(compressed_path))
    restored_bgr = restore_to_size(
        reload_compressed_bgr,
        target_size=(width, height),
        sharpen_amount=restore_sharpen,
        detail_enhance=detail_enhance,
    )
    restored_size = save_jpeg(restored_path, restored_bgr, restored_quality)

    # 4. 压缩还原模式默认不做去模糊；只有显式启用时才额外输出 deblurred_4k.jpg。
    deblurred_size = None
    psnr_deblurred = None
    ssim_deblurred = None
    blur_deblurred = None
    if deblur_mode != "none":
        deblur_processor = DeblurProcessor(
            mode=deblur_mode,
            unsharp_amount=deblur_unsharp,
        )
        deblurred_bgr = deblur_processor.apply(restored_bgr)
        deblurred_path = sample_dir / prefixed_name(filename_prefix, "deblurred_4k.jpg")
        deblurred_size = save_jpeg(deblurred_path, deblurred_bgr, restored_quality)
        psnr_deblurred = psnr(image_bgr, deblurred_bgr)
        ssim_deblurred = ssim_score(image_bgr, deblurred_bgr)
        blur_deblurred = blur_laplacian_var(deblurred_bgr)

    # 5. 计算文件大小压缩比、节省比例和图像质量指标。
    ratio = original_size / compressed_size if compressed_size > 0 else float("inf")
    saved_percent = (1.0 - compressed_size / original_size) * 100.0 if original_size > 0 else 0.0
    
    # 计算处理时间
    processing_time = time.time() - start_time

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
        compression_scale=compression_scale,
        processing_time_sec=processing_time,
    )

    write_sample_metadata(sample_dir / prefixed_name(filename_prefix, "metrics.json"), metrics)
    return metrics


def create_video_writer(
    output_path: Path,
    frame_width: int,
    frame_height: int,
    fps: float,
    codec: str = "mp4v",
) -> cv2.VideoWriter:
    """创建视频写入器。"""
    
    ensure_dir(output_path.parent)
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))
    
    if not writer.isOpened():
        raise IOError(f"Failed to create video writer: {output_path}")
    
    return writer


def process_video_to_mp4(
    input_path: Path,
    output_dir: Path,
    compression_scale: float,
    original_quality: int,
    compressed_quality: int,
    restored_quality: int,
    restore_sharpen: float,
    detail_enhance: bool,
    filename_prefix: str = "",
) -> Tuple[Path, Path, dict]:
    """处理整个视频，输出压缩和还原后的 MP4 文件。
    
    Args:
        filename_prefix: 文件名前缀，用于区分不同来源的视频（平铺模式）
    
    Returns:
        Tuple[Path, Path, dict]: (compressed_mp4_path, restored_mp4_path, metadata_dict)
    """
    
    start_time = time.time()
    
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {input_path}")
    
    # 获取视频信息
    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if source_fps <= 0.0:
        source_fps = 30.0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    orig_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 计算压缩后的尺寸
    comp_width = max(1, int(round(orig_width * compression_scale)))
    comp_height = max(1, int(round(orig_height * compression_scale)))
    
    print(f"Video info: {total_frames} frames, {source_fps:.3f} FPS, {orig_width}x{orig_height} -> {comp_width}x{comp_height}")
    
    # 创建输出文件路径（使用文件名前缀）
    if filename_prefix:
        compressed_mp4_path = output_dir / f"{filename_prefix}_compressed.mp4"
        restored_mp4_path = output_dir / f"{filename_prefix}_restored.mp4"
    else:
        compressed_mp4_path = output_dir / "compressed.mp4"
        restored_mp4_path = output_dir / "restored.mp4"
    
    # 创建视频写入器
    compressed_writer = create_video_writer(compressed_mp4_path, comp_width, comp_height, source_fps)
    restored_writer = create_video_writer(restored_mp4_path, orig_width, orig_height, source_fps)
    
    print(f"Output videos will use FPS: {source_fps:.3f} (same as source)")
    
    try:
        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            
            # 压缩帧
            compressed_frame = resize_by_scale(frame, compression_scale)
            compressed_writer.write(compressed_frame)
            
            # 还原帧
            restored_frame = restore_to_size(
                compressed_frame,
                (orig_width, orig_height),
                restore_sharpen,
                detail_enhance,
            )
            restored_writer.write(restored_frame)
            
            frame_index += 1
            if frame_index % 100 == 0:
                print(f"  Processed {frame_index}/{total_frames} frames")
    
    finally:
        capture.release()
        compressed_writer.release()
        restored_writer.release()
    
    # 计算总处理时间
    processing_time = time.time() - start_time
    print(f"✓ Processing completed in {processing_time:.2f} seconds")
    print(f"✓ Compressed video saved: {compressed_mp4_path}")
    print(f"✓ Restored video saved: {restored_mp4_path}")
    
    # 获取文件大小
    source_file_size = input_path.stat().st_size
    compressed_file_size = compressed_mp4_path.stat().st_size
    restored_file_size = restored_mp4_path.stat().st_size
    
    # 返回包含所有信息的字典
    result = {
        "input_path": str(input_path),
        "compression_scale": compression_scale,
        "original_quality": original_quality,
        "compressed_quality": compressed_quality,
        "restored_quality": restored_quality,
        "restore_sharpen": restore_sharpen,
        "detail_enhance": detail_enhance,
        "source_fps": source_fps,
        "total_frames": total_frames,
        "original_resolution": f"{orig_width}x{orig_height}",
        "compressed_resolution": f"{comp_width}x{comp_height}",
        "processing_time_sec": processing_time,
        "source_file_size": source_file_size,
        "compressed_file_size": compressed_file_size,
        "restored_file_size": restored_file_size,
    }
    
    return compressed_mp4_path, restored_mp4_path, result
