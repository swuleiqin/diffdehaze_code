# DiffDehaze

**夜间图像去雾 · 推理代码与预训练权重**

[方法概览](#方法概览) · [环境安装](#环境安装) · [预训练权重](#预训练权重) · [快速推理](#快速推理) · [致谢](#致谢)

## 方法概览

DiffDehaze 输入夜间有雾 RGB 图像，依次执行潜空间残差移除、条件扩散去噪和自适应亮度矫正。

1. **潜空间残差移除。** VAE 将有雾图编码为潜变量。残差预测器在固定噪声时间步 `t=354` 预测并移除退化残差，再利用已知的高斯噪声分量恢复清晰潜变量估计。
2. **条件扩散去噪。** 对清晰潜变量估计重新采样高斯噪声并加噪到 `t=999`。Stable Diffusion 2.1 Base 的 U-Net 加载 LoRA，配合 ControlNet 执行 **50 步**去噪。ControlNet 使用原始有雾图的潜变量作为图像条件，文本编码器提供场景条件。VAE 将去噪后的潜变量解码为 RGB 图像。
3. **自适应亮度矫正（ABAM）。** 根据输出图像的照明分布与权重内嵌的清晰夜景先验之间的 KL 散度，筛选需要矫正的图像。入选图像经过 gamma 预测和增益约束，其他图像沿用 VAE 解码结果。

## 环境安装

建议使用 Python 3.10 和 NVIDIA CUDA GPU。请在仓库根目录执行以下命令。先按显卡和驱动安装一组匹配的 PyTorch 与 torchvision，再安装仓库依赖。各版本安装命令见 [PyTorch 官方版本列表](https://pytorch.org/get-started/previous-versions/)。

```bash
conda create -n diffdehaze python=3.10 -y
conda activate diffdehaze

python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

RTX 5090 等 Blackwell 显卡可将上述 PyTorch 安装命令换成下面的 CUDA 12.8 构建。该架构的支持从 PyTorch 2.7 开始。 [PyTorch 2.7 发布说明](https://pytorch.org/blog/pytorch-2-7/)

```bash
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
```

## 预训练权重

推理需要三个任务权重和一份基础模型权重。任务权重的网盘提取码为 `6d3j`。基础模型需要单独下载。 [百度网盘 `weights` 文件夹](https://pan.baidu.com/s/1BDWfpKiDcCyXJajYSmFfew?pwd=6d3j) · [Stable Diffusion 2.1 Base 512 EMA 剪枝权重（Hugging Face 社区镜像）](https://huggingface.co/sd2-community/stable-diffusion-2-1-base/blob/main/v2-1_512-ema-pruned.ckpt)

| `configs/inference.yaml` 字段 | 模块 | 文件 |
| --- | --- | --- |
| `weights.residual` | 潜空间残差预测器 | `residual.pt` |
| `weights.stage2` | ControlNet 和 U-Net LoRA | `0012000.pt` |
| `weights.abam` | 自适应亮度矫正模块 | `abam.pt` |
| `weights.sd` | Stable Diffusion 2.1 Base 的 U-Net、VAE 和 OpenCLIP | `v2-1_512-ema-pruned.ckpt` |

`0012000.pt` 提供 ControlNet 参数和 U-Net 的 LoRA 参数，基础 U-Net、VAE 和 OpenCLIP 的参数由 `v2-1_512-ema-pruned.ckpt` 加载。`abam.pt` 内嵌推理所需的照明先验。

将网盘的 `weights/` 放在仓库根目录后，可以按下列结构放置文件。在配置中填写各文件的实际路径。

```text
weights/
├── v2-1_512-ema-pruned.ckpt
├── residual_stage14_gpt0/
│   └── checkpoints/
│       └── residual.pt
├── noise/
│   └── checkpoints/
│       └── 0012000.pt
└── abam/
    └── 不能删除multiscale_pixelmax01/
        └── checkpoints/
            └── abam.pt
```

## 快速推理

创建输入目录并放入有雾图像，支持 PNG、JPG 和 JPEG。程序读取该目录的直接子文件。在 `configs/inference.yaml` 中设置输入 `inference.image_folder`、中间图 `inference.result_folder` 和最终输出 `abam.output_folder`，或通过 `--input` 和 `--output` 指定本次运行的目录。相对路径均以仓库根目录为基准。

```bash
python inference.py --config configs/inference.yaml --check
OMP_NUM_THREADS=8 ATTN_MODE=sdp python inference.py --config configs/inference.yaml
```

`--check` 检查配置、输入和权重路径。正式推理会进一步检查权重内容。默认设备为 `cuda:0`，可用 `--device cuda:1` 指定其他 GPU。重复运行时可以指定新的输出目录，或用 `--overwrite` 允许替换已有 ABAM 结果。扩散中间图可直接覆盖。

### 文本条件

可以用 `--prompt` 为整批图像提供同一个场景描述，或用 `--caption-csv` 为每张图像提供描述。两个命令行选项互斥。配置未提供逐图 CSV 时，程序默认使用 `inference.prompt`，并将场景描述与 `inference.prompt_suffix` 拼接后输入文本编码器。

```bash
python inference.py --config configs/inference.yaml --input ./inputs --output ./results --prompt "a nighttime street with streetlights"
```

CSV 用 `image_path` 和 `caption` 两列，按去掉扩展名的文件名匹配，忽略大小写。使用 CSV 时，每张输入图像都应有一条描述。

```csv
image_path,caption
001.png,A nighttime street with streetlights
002.jpg,A building beside a dark road
```

### 输出文件

长边超过 768 像素的图像先按宽高比缩放，推理张量补齐至 64 的倍数。VAE 解码后裁掉填充并恢复输入分辨率。采用 `--output ./results` 时，扩散恢复的中间图位于 `results/stage2/`，最终图位于 `results/abam/images/`。两者均保存为以输入文件名主体命名的 PNG。ABAM 的筛选记录保存在 `results/abam/screening_report.json`。`abam.save_maps` 控制是否保存有效 gamma、照明、校正后照明和校正权重图。

## B 版配置

下载权重后，填写 `configs/inference.yaml` 中的四个 `weights` 字段。默认读取 `inputs/`，使用场景描述 `a nighttime scene`，并输出到 `results/`。ABAM 默认按 KL 阈值筛选全部图像。使用逐图描述时，通过 `--caption-csv` 指定文件。

## 许可

代码许可见 [LICENSE](LICENSE)，第三方组件声明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 致谢

扩散恢复实现基于 DiffBIR 的模型组件，使用 Stable Diffusion 2.1 Base 和 ControlNet。残差预测模块参考 DRDD 的固定噪声残差移除方法。 [DiffBIR](https://github.com/XPixelGroup/DiffBIR) · [Stable Diffusion 2.1 Base](https://huggingface.co/sd2-community/stable-diffusion-2-1-base) · [ControlNet](https://github.com/lllyasviel/ControlNet) · [DRDD](https://github.com/HKU-HealthAI/DRDD)


