"""算法模块：压缩还原、图像质量评估、运动模糊去除接口。"""

import math
from typing import Optional, Tuple

import cv2
import numpy as np

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


def motion_kernel(length: int, angle_deg: float) -> np.ndarray:
    """生成指定长度和角度的线性运动模糊卷积核。"""

    length = max(3, int(length))
    if length % 2 == 0:
        length += 1
    kernel = np.zeros((length, length), dtype=np.float32)
    center = length // 2
    kernel[center, :] = 1.0
    matrix = cv2.getRotationMatrix2D((center, center), angle_deg, 1.0)
    kernel = cv2.warpAffine(kernel, matrix, (length, length))
    kernel_sum = float(kernel.sum())
    if kernel_sum <= 1e-6:
        kernel[center, center] = 1.0
        kernel_sum = 1.0
    return kernel / kernel_sum


def wiener_deconvolution(
    image_bgr: np.ndarray,
    kernel: np.ndarray,
    noise_power: float = 0.02,
) -> np.ndarray:
    """传统 Wiener 去卷积，用于调试已知方向和长度的运动模糊。"""

    image = image_bgr.astype(np.float32) / 255.0
    height, width = image.shape[:2]
    padded_kernel = np.zeros((height, width), dtype=np.float32)
    kh, kw = kernel.shape[:2]
    padded_kernel[:kh, :kw] = kernel
    padded_kernel = np.roll(padded_kernel, -kh // 2, axis=0)
    padded_kernel = np.roll(padded_kernel, -kw // 2, axis=1)
    kernel_fft = np.fft.fft2(padded_kernel)
    kernel_power = np.abs(kernel_fft) ** 2
    inverse_filter = np.conj(kernel_fft) / (kernel_power + noise_power)

    channels = []
    for channel_index in range(3):
        channel_fft = np.fft.fft2(image[:, :, channel_index])
        restored = np.fft.ifft2(channel_fft * inverse_filter).real
        channels.append(restored)
    merged = np.stack(channels, axis=2)
    return np.clip(merged * 255.0, 0, 255).astype(np.uint8)


class DeblurProcessor:
    """运动模糊去除的统一调试接口。

    模式：
      - none：不处理，直接返回输入图。
      - unsharp：快速锐化基线，适合先验证后处理链路。
      - wiener：传统运动核 Wiener 去卷积。

    后续如需接入深度学习模型，可以只替换 apply() 内部逻辑。
    """

    def __init__(
        self,
        mode: str,
        motion_length: int,
        motion_angle: float,
        wiener_noise: float,
        unsharp_amount: float,
    ) -> None:
        """保存去模糊参数，便于 CLI 调试不同模式和不同运动核。"""

        self.mode = mode
        self.motion_length = motion_length
        self.motion_angle = motion_angle
        self.wiener_noise = wiener_noise
        self.unsharp_amount = unsharp_amount

    def apply(self, image_bgr: np.ndarray) -> np.ndarray:
        """根据 deblur_mode 执行对应去模糊策略。"""

        if self.mode == "none":
            return image_bgr
        if self.mode == "unsharp":
            return unsharp_mask(image_bgr, amount=self.unsharp_amount, sigma=1.2)
        if self.mode == "wiener":
            kernel = motion_kernel(self.motion_length, self.motion_angle)
            return wiener_deconvolution(image_bgr, kernel, noise_power=self.wiener_noise)
        raise ValueError(f"Unsupported deblur mode: {self.mode}")
