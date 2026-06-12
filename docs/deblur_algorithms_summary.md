# 去模糊算法调研汇总

本文整理此前尝试和讨论过的去模糊相关算法，重点面向消化内镜视频场景。结论先行：目前你的实验反馈里，`CDVD-TSP` 效果最好，说明当前任务更依赖前后帧时序信息，而不是单图自然图像先验。

排序规则：

- `CDVD-TSP` 单独置顶，因为它是当前实验效果最好的方法。
- 其余算法按照论文会议/出处时间排序。

## 总体判断

| 序号 | 算法 | 输入类型 | 技术路线 | 会议/出处 | 实验备注 |
|---:|---|---|---|---|---|
| 1 | CDVD-TSP | 多帧视频 | 光流 + 级联恢复 + temporal sharpness prior | CVPR 2020 | 当前实验最好，说明多帧时序信息对胃肠镜视频更关键 |
| 2 | DeblurGAN-v2 | 单图 | GAN + FPN generator | ICCV 2019 | 效果很弱，几乎不改变原图 |
| 3 | ESTRNN | 多帧视频 | 轻量 RNN + 时空注意力 | ECCV 2020 Spotlight / IJCV | 伪影和色调改变都比较严重 |
| 4 | MPRNet | 单图 | multi-stage progressive restoration | CVPR 2021 | 几乎不变 |
| 5 | Restormer | 单图 | efficient transformer restoration | CVPR 2022 Oral | 会较大改变原图结构 |
| 6 | NAFNet | 单图 | activation-free CNN baseline | ECCV 2022 | 几乎不变 |
| 7 | Stripformer | 单图 | horizontal/vertical strip attention | ECCV 2022 Oral | 只有色调上的一些小变化 |
| 8 | RVRT | 多帧视频 | recurrent video restoration transformer + deformable attention | NeurIPS 2022 | 变化不大，速度很慢，内存占用很高 |
| 9 | FFTformer | 单图 | frequency-domain transformer | CVPR 2023 | 已尝试，整体不如 CDVD-TSP 方向 |
| 10 | Blur2Blur | 单图训练策略 | unknown-domain 无监督 blur-to-blur 适配 | CVPR 2024 | 没有 checkpoint，没法直接推理，跳过 |
| 11 | MISCFilter | 单图 | motion-adaptive separable collaborative filter | CVPR 2024 | 格子化效应很大，效果最差 |
| 12 | AdaIR | 单图通用恢复 | all-in-one + frequency mining/modulation | ICLR 2025 | 集成 5 种增强功能，结果很差，每帧之间不稳定 |
| 13 | EVSSM | 单图 | visual state space model | CVPR 2025 / AIM 2025 | 部分帧出现严重的彩虹伪影 |
| 14 | EndoCaver | 单图内镜 | 内镜 deblur + segmentation 双任务 | ICASSP 2026 | 自己训练后效果依然差 |

## 技术路线分组

### 多帧视频去模糊

这组最值得继续围绕内镜视频深入，因为它利用前后帧信息。当前实验中，`CDVD-TSP` 明显比大多数单图模型更有价值。

| 算法 | 核心思路 | 实现手段 | 实验备注 |
|---|---|---|---|
| CDVD-TSP | 相邻帧里存在更清晰的时序线索 | 估计中间 latent frame 光流，级联恢复，并用 temporal sharpness prior 约束恢复 | 当前实验最好，和你的 `selected` 选帧思路高度一致 |
| ESTRNN | 用过去/未来帧帮助恢复当前帧 | recurrent cell + residual dense block + global spatio-temporal attention | 伪影和色调改变都比较严重 |
| RVRT | 在 recurrent 框架里融合局部 clip 信息 | recurrent transformer + guided deformable attention | 变化不大，速度很慢，内存占用很高 |

### 单图自然图像去模糊

这些模型大多在 GoPro、REDS、RealBlur、HIDE 等自然图像数据上训练。它们知名度和代码成熟度高，但你的实验已经说明，直接迁移到胃肠镜视频帧通常收益有限。

| 算法 | 核心思路 | 实现手段 | 实验备注 |
|---|---|---|---|
| DeblurGAN-v2 | 用 GAN 提升感知锐度和速度 | FPN generator + double-scale discriminator | 效果很弱，几乎不改变原图 |
| MPRNet | 分阶段逐步恢复 | 多阶段 encoder-decoder + supervised attention | 几乎不变 |
| Restormer | 高分辨率图像恢复 Transformer | MDTA attention + gated-Dconv FFN | 会较大改变原图结构 |
| NAFNet | 简化 CNN，去掉常规非线性激活 | Nonlinear Activation Free block + U-Net 式结构 | 几乎不变 |
| Stripformer | 建模横向/纵向条带模糊 | intra-strip/inter-strip attention | 只有色调上的一些小变化 |
| FFTformer | 在频域降低 Transformer 计算成本 | frequency-domain self-attention + frequency FFN | 已尝试，整体不如 CDVD-TSP 方向 |
| MISCFilter | 在图像空间显式处理空间变化运动 | motion estimation network 预测 flow/mask/kernel/offset/weight，再协同滤波 | 格子化效应很大，效果最差 |
| EVSSM | 用状态空间模型捕获长程依赖 | efficient visual scan block + SSM | 部分帧出现严重的彩虹伪影 |

### 内镜/未知域/通用恢复

这组不是传统 GoPro 路线，更贴近“域差异”问题。但从当前实验看，单图内镜恢复或通用恢复仍没有解决胃肠镜视频运动模糊问题。

| 算法 | 核心思路 | 实现手段 | 实验备注 |
|---|---|---|---|
| Blur2Blur | unknown-domain 下用未配对数据适配去模糊 | 把难去除 blur 转成更易去除 blur，再训练 deblur | 没有 checkpoint，没法直接推理，跳过 |
| AdaIR | 一个模型处理多退化 | 频率挖掘和调制，覆盖 denoise/dehaze/derain/deblur/enhancement | 集成 5 种增强功能，结果很差，每帧之间不稳定 |
| EndoCaver | 内镜图像同时做 deblur 和 segmentation | 轻量 transformer，双 decoder，restored RGB + polyp mask | 自己训练后效果依然差 |

## 论文与仓库明细

| 算法 | 论文名 | 作者 | 会议/出处 | GitHub | 实验备注 |
|---|---|---|---|---|---|
| CDVD-TSP | Cascaded Deep Video Deblurring Using Temporal Sharpness Prior | Jinshan Pan, Haoran Bai, Jinhui Tang | CVPR 2020 | https://github.com/csbhr/CDVD-TSP | 当前实验最好 |
| DeblurGAN-v2 | DeblurGAN-v2: Deblurring (Orders-of-Magnitude) Faster and Better | Orest Kupyn, Tetiana Martyniuk, Junru Wu, Zhangyang Wang | ICCV 2019 | https://github.com/VITA-Group/DeblurGANv2 | 效果很弱，几乎不改变原图 |
| ESTRNN | Real-world Video Deblurring: A Benchmark Dataset and An Efficient Recurrent Neural Network | Zhihang Zhong, Ye Gao, Yinqiang Zheng, Bo Zheng, Imari Sato | ECCV 2020 Spotlight / IJCV | https://github.com/zzh-tech/ESTRNN | 伪影和色调改变都比较严重 |
| MPRNet | Multi-Stage Progressive Image Restoration | Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, Ling Shao | CVPR 2021 | https://github.com/swz30/MPRNet | 几乎不变 |
| Restormer | Restormer: Efficient Transformer for High-Resolution Image Restoration | Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang | CVPR 2022 Oral | https://github.com/swz30/Restormer | 会较大改变原图结构 |
| NAFNet | Simple Baselines for Image Restoration | Liangyu Chen, Xiaojie Chu, Xiangyu Zhang, Jian Sun | ECCV 2022 | https://github.com/megvii-research/NAFNet | 几乎不变 |
| Stripformer | Stripformer: Strip Transformer for Fast Image Deblurring | Fu-Jen Tsai, Yan-Tsung Peng, Yen-Yu Lin, Chung-Chi Tsai, Chia-Wen Lin | ECCV 2022 Oral | https://github.com/pp00704831/Stripformer-ECCV-2022- | 只有色调上的一些小变化 |
| RVRT | Recurrent Video Restoration Transformer with Guided Deformable Attention | Jingyun Liang, Yuchen Fan, Xiaoyu Xiang, Rakesh Ranjan, Eddy Ilg, Simon Green, Jiezhang Cao, Kai Zhang, Radu Timofte, Luc Van Gool | NeurIPS 2022 | https://github.com/JingyunLiang/RVRT | 变化不大，速度很慢，内存占用很高 |
| FFTformer | Efficient Frequency Domain-based Transformers for High-Quality Image Deblurring | Lingshun Kong, Jiangxin Dong, Mingqiang Li, Jianjun Ge, Jinshan Pan | CVPR 2023 | https://github.com/kkkls/FFTformer | 已尝试，整体不如 CDVD-TSP 方向 |
| Blur2Blur | Blur2Blur: Blur Conversion for Unsupervised Image Deblurring on Unknown Domains | Bang-Dang Pham, Phong Tran, Anh Tran, Cuong Pham, Rang Nguyen, Minh Hoai | CVPR 2024 | https://github.com/VinAIResearch/Blur2Blur | 没有 checkpoint，没法直接推理，跳过 |
| MISCFilter | Motion-adaptive Separable Collaborative Filters for Blind Motion Deblurring | Chengxu Liu, Xuan Wang, Xiangyu Xu, Ruhao Tian, Shuai Li, Xueming Qian, Ming-Hsuan Yang | CVPR 2024 | https://github.com/ChengxuLiu/MISCFilter | 格子化效应很大，效果最差 |
| AdaIR | AdaIR: Adaptive All-in-One Image Restoration via Frequency Mining and Modulation | Yuning Cui, Syed Waqas Zamir, Salman Khan, Alois Knoll, Mubarak Shah, Fahad Shahbaz Khan | ICLR 2025 | https://github.com/c-yn/AdaIR | 集成 5 种增强功能，结果很差，每帧之间不稳定 |
| EVSSM | Efficient Visual State Space Model for Image Deblurring | Lingshun Kong, Jiangxin Dong, Ming-Hsuan Yang, Jinshan Pan | CVPR 2025 / AIM 2025 High FPS Motion Deblurring | https://github.com/kkkls/EVSSM | 部分帧出现严重的彩虹伪影 |
| EndoCaver | EndoCaver: Handling Fog, Blur and Glare in Endoscopic Images via Joint Deblurring-Segmentation | Zhuoyu Wu, Wenhui Ou, Pei-Sze Tan, Jiayan Yang, Wenqi Fang, Zheng Wang, Raphael C.-W. Phan | ICASSP 2026 | https://github.com/ReaganWu/EndoCaver | 自己训练后效果依然差 |

## 作者与团队同源关系

### Zamir / Khan 图像恢复线

这条线包括：

- MPRNet
- Restormer
- AdaIR

共同特点：

- 都是图像恢复领域强基线。
- 关注多退化恢复、Transformer 或频率建模。
- 在自然图像 benchmark 上很强，但直接迁移到胃肠镜视频效果不理想。

作者重叠：

- Syed Waqas Zamir：MPRNet、Restormer、AdaIR
- Salman Khan：MPRNet、Restormer、AdaIR
- Fahad Shahbaz Khan：MPRNet、Restormer、AdaIR
- Ming-Hsuan Yang：MPRNet、Restormer

你的实验备注：

- MPRNet 几乎不变。
- Restormer 会较大改变原图结构。
- AdaIR 结果很差，且每帧之间不稳定。

### Jinshan Pan / Lingshun Kong 去模糊线

这条线包括：

- CDVD-TSP
- FFTformer
- EVSSM

共同特点：

- 更聚焦 deblurring。
- CDVD-TSP 是视频多帧时序先验。
- FFTformer/EVSSM 是后续单图高效去模糊路线。

作者重叠：

- Jinshan Pan：CDVD-TSP、FFTformer、EVSSM
- Lingshun Kong：FFTformer、EVSSM
- Jiangxin Dong：FFTformer、EVSSM

你的实验里 CDVD-TSP 表现更好，说明这个团队的“视频时序先验”路线比其后续单图路线更贴近内镜视频。EVSSM 虽然更新，但部分帧出现严重彩虹伪影。

### Ming-Hsuan Yang 相关线

Ming-Hsuan Yang 参与或共同作者出现在：

- MPRNet
- Restormer
- MISCFilter
- EVSSM

这说明多个高质量恢复/去模糊方法之间存在研究网络重叠，但技术路线不同：`MISCFilter` 偏真实盲运动滤波，`EVSSM` 偏 SSM 单图建模，`MPRNet/Restormer` 偏通用图像恢复。你的实验里，`MISCFilter` 格子化效应很大，`EVSSM` 有彩虹伪影，`Restormer` 改变结构明显。

### 内镜专用线

目前列表里真正直接针对内镜图像的是：

- EndoCaver

它处理 fog、blur、glare，并联合 segmentation。但它是单图模型，不利用前后帧。你的实验里，即使自己训练后效果依然差。对胃肠镜视频，后续更值得关注“内镜视频质量评估 + 多帧恢复”而不是单图 restoration。

## 对当前项目的建议

从实验结果看，后续不建议继续大范围尝试单图通用模型。更合理的方向是：

1. 以 CDVD-TSP 作为外部强基线。
2. 保留当前 `current -> selected -> deblur` 的可解释流程。
3. 把整帧 selected 升级为局部/patch 级 selected 或多帧加权融合。
4. 如果后续训练，优先做内镜域的多帧或 temporal sharpness 适配，而不是从零训练一个单图通用模型。

一句话总结：

```text
内镜视频模糊问题更像“时序信息选择与融合”，而不只是“单张图片复原”。
```

## 参考链接

- CDVD-TSP: https://github.com/csbhr/CDVD-TSP
- DeblurGAN-v2: https://github.com/VITA-Group/DeblurGANv2
- ESTRNN: https://github.com/zzh-tech/ESTRNN
- MPRNet: https://github.com/swz30/MPRNet
- Restormer: https://github.com/swz30/Restormer
- NAFNet: https://github.com/megvii-research/NAFNet
- Stripformer: https://github.com/pp00704831/Stripformer-ECCV-2022-
- RVRT: https://github.com/JingyunLiang/RVRT
- FFTformer: https://github.com/kkkls/FFTformer
- Blur2Blur: https://github.com/VinAIResearch/Blur2Blur
- MISCFilter: https://github.com/ChengxuLiu/MISCFilter
- AdaIR: https://github.com/c-yn/AdaIR
- EVSSM: https://github.com/kkkls/EVSSM
- EndoCaver: https://github.com/ReaganWu/EndoCaver
