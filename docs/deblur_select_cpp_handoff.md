# deblur_select C++ 对接复现说明

本文面向 C++ 端开发人员，目标是快速复现 `deblur_select` 的核心逻辑。C++ 端不需要照搬 Python 的 UI 和目录批处理，只需要实现下面几个核心函数和调用顺序。

## 术语约定

| 名称 | 含义 |
|---|---|
| `current` | 用户按键时的视频当前帧，也就是中心帧。 |
| `selected` | 在 `current` 前后邻域内，用清晰度分数选出的最佳帧。 |
| `deblur` | 对 `selected` 做亮度通道 unsharp 后得到的最终输出。 |

核心流程：

```text
用户按 S
  ↓
current = 当前视频帧
  ↓
在 current 前后 temporal_radius 帧范围内按 temporal_stride 搜索候选帧
  ↓
对每个候选帧计算 endoscopy_sharpness_score
  ↓
score 最高者成为 selected
  ↓
对 selected 做 luminance_unsharp_mask
  ↓
保存 current / selected / deblur / metrics
```

## C++ 端最小实现函数清单

| 优先级 | Python 函数 | 文件 | C++ 端是否必须复现 | 作用 |
|---:|---|---|---|---|
| 1 | `endoscopy_sharpness_score()` | `algorithms.py` | 必须 | 计算内镜帧清晰度，用于选 `selected`。 |
| 2 | `select_best_frame_in_window()` | `algorithms.py` | 必须 | 在当前帧邻域中寻找 score 最高的 `selected`。 |
| 3 | `luminance_unsharp_mask()` | `algorithms.py` | 必须 | 对 `selected` 的 LAB 亮度通道做 unsharp，生成 `deblur`。 |
| 4 | `blur_laplacian_var()` | `algorithms.py` | 建议 | 调试指标，用于输出 blur score。 |
| 5 | `DeblurProcessor.apply()` | `algorithms.py` | 可简化 | Python 的模式分发器，C++ 可直接调用对应函数。 |
| 6 | `process_video_interactive()` 中按 `S` 的保存逻辑 | `processing.py` | 参考 | 串起 current、selected、deblur 和 metadata。 |
| 7 | `DeblurSelectionRecord` | `summary.py` | 参考 | 定义保存的 metrics 字段。 |

不需要优先复现的部分：

- OpenCV 调试窗口布局。
- 按键 UI 文本。
- README / CSV 汇总。
- 压缩还原 `compress_restore` 逻辑。

## 1. 清晰度评分：endoscopy_sharpness_score

Python 位置：

```text
algorithms.py::endoscopy_sharpness_score(image_bgr)
```

输入：

```text
BGR uint8 image
```

输出：

```text
double score
```

算法意图：

- 排除黑边。
- 排除强反光。
- 用 Sobel 梯度统计衡量清晰度。
- 同时看整体平均梯度和高分位梯度，避免只被少量噪声点影响。

伪代码：

```cpp
double endoscopySharpnessScore(const cv::Mat& bgr) {
    cv::Mat gray;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);

    // 有效区域：排除黑边和强反光。
    cv::Mat validMask = (gray > 18) & (gray < 245);
    if (cv::countNonZero(validMask) < 256) {
        validMask = gray > 0;
    }

    cv::Mat smoothed;
    cv::GaussianBlur(gray, smoothed, cv::Size(3, 3), 0);

    cv::Mat gradX, gradY;
    cv::Sobel(smoothed, gradX, CV_32F, 1, 0, 3);
    cv::Sobel(smoothed, gradY, CV_32F, 0, 1, 3);

    cv::Mat magnitude;
    cv::magnitude(gradX, gradY, magnitude);

    std::vector<float> values;
    values.reserve(cv::countNonZero(validMask));
    for (int y = 0; y < magnitude.rows; ++y) {
        const float* magRow = magnitude.ptr<float>(y);
        const uchar* maskRow = validMask.ptr<uchar>(y);
        for (int x = 0; x < magnitude.cols; ++x) {
            if (maskRow[x]) {
                values.push_back(magRow[x]);
            }
        }
    }

    if (values.empty()) {
        return 0.0;
    }

    double meanScore = mean(values);
    double highScore = percentile(values, 90.0);
    return 0.65 * highScore + 0.35 * meanScore;
}
```

注意：

- `percentile(values, 90.0)` 可以通过排序或 `std::nth_element` 实现。
- Python 当前实现里 `valid_mask` 的规则是 `gray > 18 && gray < 245`。
- 分数只是相对比较用，不要求跨视频绝对一致。

## 2. 邻域选帧：select_best_frame_in_window

Python 位置：

```text
algorithms.py::select_best_frame_in_window(...)
```

输入：

| 参数 | 含义 |
|---|---|
| `center_index` | `current` 的 0-based 帧号。 |
| `frame_reader(index)` | 读取指定帧，返回图像和时间戳。 |
| `total_frames` | 视频总帧数。 |
| `video_fps` | 视频 FPS。 |
| `mode` | `unsharp` 或 `temporal_unsharp`。 |
| `temporal_radius` | 向前/向后搜索多少帧。 |
| `temporal_stride` | 候选帧采样步长。 |

输出：

```text
selected_frame
selected_index
selected_timestamp
selected_score
selected_offset = selected_index - center_index
```

伪代码：

```cpp
SelectedResult selectBestFrameInWindow(
    int centerIndex,
    int totalFrames,
    double fps,
    const std::string& mode,
    int temporalRadius,
    int temporalStride,
    std::function<std::pair<cv::Mat, double>(int)> readFrame
) {
    if (mode != "temporal_unsharp") {
        auto [frame, timestamp] = readFrame(centerIndex);
        double score = endoscopySharpnessScore(frame);
        return {frame, centerIndex, timestamp, score, 0};
    }

    int radius = std::max(0, temporalRadius);
    int stride = std::max(1, temporalStride);
    int firstIndex = std::max(0, centerIndex - radius);
    int lastIndex = std::min(std::max(totalFrames - 1, 0), centerIndex + radius);

    cv::Mat bestFrame;
    int bestIndex = centerIndex;
    double bestTimestamp = centerIndex / fps;
    double bestScore = -1.0;

    for (int i = firstIndex; i <= lastIndex; i += stride) {
        auto [candidateFrame, candidateTimestamp] = readFrame(i);
        double score = endoscopySharpnessScore(candidateFrame);
        if (score > bestScore) {
            bestFrame = candidateFrame.clone();
            bestIndex = i;
            bestTimestamp = candidateTimestamp;
            bestScore = score;
        }
    }

    return {
        bestFrame,
        bestIndex,
        bestTimestamp,
        bestScore,
        bestIndex - centerIndex
    };
}
```

关键点：

- Python 和 C++ 都使用 0-based 帧号。
- UI 显示时可以用 `frame_index + 1`。
- 如果 `mode == "unsharp"`，不做邻域搜索，`selected = current`。
- 如果 `mode == "temporal_unsharp"`，才搜索邻域。

## 3. 性能警示：不要对每个候选帧随机 seek

最直观的实现是：

```text
for candidate in temporal_window:
    capture.set(CAP_PROP_POS_FRAMES, candidate)
    capture.read()
    score(candidate)
```

这个写法在压缩视频上会很慢。每次随机 seek 都可能触发解码器从附近关键帧重新解码，实际耗时会明显高于清晰度评分本身。

Python 端已经做过优化，耗时明显下降。C++ 端建议也采用同样策略：

```text
按 S 后：
  1. 计算 first_index / last_index / candidate_indices
  2. capture.set(CAP_PROP_POS_FRAMES, first_index)
  3. 从 first_index 顺序 read 到 last_index
  4. 只把 candidate_indices 中的帧放入缓存
  5. selectBestFrameInWindow 从缓存读取候选帧并评分
```

C++ 伪代码：

```cpp
std::unordered_map<int, CachedFrame> prefetchTemporalWindow(
    cv::VideoCapture& capture,
    int centerIndex,
    const cv::Mat& currentFrame,
    double currentTimestamp,
    int totalFrames,
    double fps,
    int temporalRadius,
    int temporalStride
) {
    std::unordered_map<int, CachedFrame> cache;
    cache[centerIndex] = {currentFrame.clone(), currentTimestamp};

    int radius = std::max(0, temporalRadius);
    int stride = std::max(1, temporalStride);
    int firstIndex = std::max(0, centerIndex - radius);
    int lastIndex = std::min(std::max(totalFrames - 1, 0), centerIndex + radius);

    std::unordered_set<int> candidates;
    for (int i = firstIndex; i <= lastIndex; i += stride) {
        candidates.insert(i);
    }

    capture.set(cv::CAP_PROP_POS_FRAMES, firstIndex);
    for (int i = firstIndex; i <= lastIndex; ++i) {
        cv::Mat frame;
        if (!capture.read(frame) || frame.empty()) {
            break;
        }
        if (candidates.count(i)) {
            cache[i] = {frame.clone(), i / fps};
        }
    }

    return cache;
}
```

然后把 `selectBestFrameInWindow()` 里的 `readFrame(index)` 实现为优先读缓存：

```cpp
auto readFrame = [&](int index) -> std::pair<cv::Mat, double> {
    auto it = cache.find(index);
    if (it != cache.end()) {
        return {it->second.frame, it->second.timestamp};
    }

    // 兜底路径：正常 seek + read。理论上不应频繁走到这里。
    capture.set(cv::CAP_PROP_POS_FRAMES, index);
    cv::Mat frame;
    capture.read(frame);
    return {frame, index / fps};
};
```

这不会改变 `selected` 的选择结果，因为候选范围、步长和清晰度评分都没有变，只是把读取方式从“多次随机 seek”改成“一次顺序解码 + 缓存”。

## 4. 亮度通道锐化：luminance_unsharp_mask

Python 位置：

```text
algorithms.py::luminance_unsharp_mask(image_bgr, amount, sigma)
```

输入：

| 参数 | 含义 |
|---|---|
| `image_bgr` | 通常是 `selected`。 |
| `amount` | 锐化强度，对应 CLI `--deblur-unsharp`。 |
| `sigma` | 高斯模糊 sigma，单位是像素，对应 CLI `--deblur-sigma`。 |

输出：

```text
deblur image, BGR uint8
```

伪代码：

```cpp
cv::Mat luminanceUnsharpMask(const cv::Mat& bgr, double amount, double sigma) {
    cv::Mat lab;
    cv::cvtColor(bgr, lab, cv::COLOR_BGR2Lab);

    std::vector<cv::Mat> channels;
    cv::split(lab, channels);
    cv::Mat& L = channels[0];

    cv::Mat blurred;
    cv::GaussianBlur(L, blurred, cv::Size(0, 0), sigma, sigma);

    cv::Mat sharpenedL;
    cv::addWeighted(L, 1.0 + amount, blurred, -amount, 0.0, sharpenedL);

    channels[0] = sharpenedL;
    cv::Mat sharpenedLab;
    cv::merge(channels, sharpenedLab);

    cv::Mat outBgr;
    cv::cvtColor(sharpenedLab, outBgr, cv::COLOR_Lab2BGR);
    return outBgr;
}
```

注意：

- Python 使用 `cv2.COLOR_BGR2LAB` / `cv2.COLOR_LAB2BGR`。
- C++ OpenCV 名称通常是 `cv::COLOR_BGR2Lab` / `cv::COLOR_Lab2BGR`。
- `cv::addWeighted` 对 `uint8` 输出会做饱和裁剪，等价于 Python 后续 `np.clip(..., 0, 255).astype(np.uint8)`。
- 只锐化 L 通道是为了减少彩色边缘伪影。

## 5. Blur 指标：blur_laplacian_var

Python 位置：

```text
algorithms.py::blur_laplacian_var(image_bgr)
```

用途：

- 调试指标。
- 数值越高通常代表边缘越丰富。
- 当前保存 `current_blur_score`、`selected_blur_score`、`deblur_blur_score`。

伪代码：

```cpp
double blurLaplacianVar(const cv::Mat& bgr) {
    cv::Mat gray;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);

    cv::Mat lap;
    cv::Laplacian(gray, lap, CV_64F);

    cv::Scalar mean, stddev;
    cv::meanStdDev(lap, mean, stddev);
    return stddev[0] * stddev[0];
}
```

注意：

- 这个指标只用于展示和记录，不参与选帧。
- 选帧用的是 `endoscopy_sharpness_score`。

## 6. 按 S 后的主流程

Python 位置：

```text
processing.py::process_video_interactive(...)
```

C++ 端可按下面顺序实现：

```cpp
// 用户按 S 时
current = currentFrame.clone();
currentIndex = frameIndex;
currentTimestamp = currentIndex / fps;

selected = selectBestFrameInWindow(
    currentIndex,
    totalFrames,
    fps,
    deblurMode,
    temporalRadius,
    temporalStride,
    readFrame
);

deblur = luminanceUnsharpMask(
    selected.frame,
    deblurUnsharp,
    deblurSigma
);

currentSharp = endoscopySharpnessScore(current);
selectedSharp = selected.score;

currentBlur = blurLaplacianVar(current);
selectedBlur = blurLaplacianVar(selected.frame);
deblurBlur = blurLaplacianVar(deblur);

save current as "__current.jpg";
save selected as "__selected.jpg";
save deblur as "__deblur.jpg";
save metrics json;
```

性能建议：

- `current / selected / deblur` 三张图互相独立，可以并行 `imwrite`。
- `current` 的 sharp/blur 指标如果在 UI 刷新时已经计算过，可以复用，避免按键后重复计算。
- 计时建议拆成两个字段：
  - `select_elapsed_sec`：从按键后开始预读窗口、评分、选出 `selected` 的耗时。
  - `total_elapsed_sec`：从按键后开始，到三张图保存和指标计算完成的耗时。

保存文件命名建议保持一致：

```text
save_000000_cur_f001001_sel_f001004_t0033.367s__current.jpg
save_000000_cur_f001001_sel_f001004_t0033.367s__selected.jpg
save_000000_cur_f001001_sel_f001004_t0033.367s__deblur.jpg
save_000000_cur_f001001_sel_f001004_t0033.367s__deblur_metrics.json
```

其中：

- `cur_f001001` 是 current 的 1-based 帧号。
- `sel_f001004` 是 selected 的 1-based 帧号。
- `t0033.367s` 是 current 时间戳。

## 7. 参数说明

| 参数 | 默认值 | C++ 端含义 |
|---|---:|---|
| `deblur_mode` | `temporal_unsharp` | `unsharp` 不搜邻域；`temporal_unsharp` 搜邻域。 |
| `temporal_radius` | `6` | 从 current 向前/向后各搜索多少帧。 |
| `temporal_stride` | `1` | 候选帧步长。 |
| `deblur_unsharp` | `0.55` | L 通道 unsharp 加回细节的强度。 |
| `deblur_sigma` | `1.2` | GaussianBlur 的 sigma，单位是像素。 |
| `frame_quality` | `95` | current 和 selected JPEG 质量。 |
| `deblur_quality` | `95` | deblur JPEG 质量。 |

推荐先对齐的参数：

```text
deblur_mode = temporal_unsharp
temporal_radius = 6
temporal_stride = 1
deblur_unsharp = 0.55
deblur_sigma = 1.2
```

## 8. Metrics JSON 字段

Python 位置：

```text
summary.py::DeblurSelectionRecord
```

C++ 端建议至少保存：

| 字段 | 含义 |
|---|---|
| `sample_id` | 输出前缀。 |
| `source_path` | 输入视频路径。 |
| `frame_index` | current 的 0-based 帧号。 |
| `timestamp_sec` | current 时间戳。 |
| `width` / `height` | 帧尺寸。 |
| `deblur_mode` | 当前模式。 |
| `current_blur_score` | current 的 Laplacian variance。 |
| `selected_blur_score` | selected 的 Laplacian variance。 |
| `deblur_blur_score` | deblur 的 Laplacian variance。 |
| `deblur_vs_selected_blur_gain` | `deblur_blur_score - selected_blur_score`。 |
| `current_sharpness_score` | current 的内镜清晰度分数。 |
| `selected_frame_index` | selected 的 0-based 帧号。 |
| `selected_timestamp_sec` | selected 时间戳。 |
| `selected_sharpness_score` | selected 的内镜清晰度分数。 |
| `selected_offset` | `selected_frame_index - frame_index`。 |
| `temporal_radius` | 搜索半径。 |
| `temporal_stride` | 搜索步长。 |
| `select_elapsed_sec` | 选帧耗时，包括窗口预读和候选帧评分。 |
| `total_elapsed_sec` | 从按键到图像保存、指标计算完成的全流程耗时。 |

## 9. C++ 复现验收标准

建议先不用接 UI，直接做一个命令行测试程序：

```text
input video
center frame index
temporal_radius
temporal_stride
deblur_unsharp
deblur_sigma
```

输出：

```text
current.jpg
selected.jpg
deblur.jpg
metrics.json
```

验收方式：

1. 对同一视频、同一 `center frame index`，C++ 选出的 `selected_frame_index` 应和 Python 一致，或在同分/近似分数时非常接近。
2. `selected_sharpness_score` 与 Python 基本一致。
3. `deblur.jpg` 肉眼应和 Python 输出一致。
4. `current/selected/deblur` 命名和含义必须保持一致。

## 10. 当前 Python 文件对应关系

| 文件 | C++ 端参考价值 |
|---|---|
| `algorithms.py` | 最高。核心算法函数都在这里。 |
| `processing.py` | 中等。主要看按 `S` 后的调用顺序和保存逻辑。 |
| `summary.py` | 中等。主要看 metrics 字段。 |
| `demo.py` | 低。主要看 CLI 参数默认值。 |
| `README.md` | 低。面向使用者，不适合直接复现算法。 |

一句话给 C++ 工程师：

```text
先复现 algorithms.py 里的 endoscopy_sharpness_score、select_best_frame_in_window、luminance_unsharp_mask，再按 processing.py 里按 S 的顺序串起来。
```
