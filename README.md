# 基于扩散模型的夜间图像去雾方法

[![预训练权重](https://img.shields.io/badge/%E9%A2%84%E8%AE%AD%E7%BB%83%E6%9D%83%E9%87%8D-%E7%99%BE%E5%BA%A6%E7%BD%91%E7%9B%98-ffd700)](https://pan.baidu.com/s/1BDWfpKiDcCyXJajYSmFfew?pwd=6d3j)
[![Stable Diffusion 2.1 Base](https://img.shields.io/badge/%E5%9F%BA%E7%A1%80%E6%A8%A1%E5%9E%8B-Stable%20Diffusion%202.1-1bb41b)](https://huggingface.co/sd2-community/stable-diffusion-2-1-base/blob/main/v2-1_512-ema-pruned.ckpt)

本仓库提供以下论文的 PyTorch 推理代码。

> **基于扩散模型的夜间图像去雾方法**<br>
> **作者**<br>
> **单位**

![夜间图像去雾方法的整体框架](.assets/overview.png)

残差预测器在固定噪声时间步 `t=354` 去除潜空间退化残差，得到清晰潜变量估计。将估计重新加噪到 `t=999` 后，使用 Stable Diffusion 2.1 Base 与 ControlNet 执行 50 步去噪。VAE 解码为 RGB 图像后，ABAM 根据清晰夜景亮度先验筛选并矫正需要提亮的图像。

⭐ 如果本项目对你的研究有帮助，欢迎为仓库点亮 Star。

## 🏃 TODO

- [x] 提供推理代码和预训练权重
- [x] 提供去雾结果示例和定量比较
- [ ] 论文链接
- [ ] 训练代码

## 🔧 依赖与安装

1. **获取代码**

   从仓库页面选择 **Code → Download ZIP**，解压后进入仓库根目录。

2. **安装依赖**

   ```bash
   conda create -n diffdehaze python=3.10 -y
   conda activate diffdehaze
   python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
   python -m pip install -r requirements.txt
   ```

   上例使用 CUDA 12.1 版 PyTorch。RTX 5090 等 Blackwell 显卡可安装 PyTorch 2.7.1、torchvision 0.22.1 和 CUDA 12.8 构建。 [PyTorch 安装命令](https://pytorch.org/get-started/previous-versions/)

## 🚀 快速推理

### 第一步　下载预训练权重

下载三个任务权重，解压后保留 `weights/` 目录结构。另行下载 Stable Diffusion 2.1 Base 512 的 `v2-1_512-ema-pruned.ckpt`，其中包含推理所需的基础 U-Net、VAE 和 OpenCLIP。ABAM 权重内嵌亮度先验。

![权重一览。残差预测器 residual.pt、ControlNet 与 U-Net LoRA 0012000.pt、ABAM abam.pt 来自百度网盘，Stable Diffusion 2.1 Base 权重来自 Hugging Face。](.assets/checkpoints.png)

[任务权重 · 百度网盘（提取码 6d3j）](https://pan.baidu.com/s/1BDWfpKiDcCyXJajYSmFfew?pwd=6d3j) · [基础模型 · Hugging Face 社区镜像](https://huggingface.co/sd2-community/stable-diffusion-2-1-base/blob/main/v2-1_512-ema-pruned.ckpt)

按本地目录填写 `configs/inference.yaml` 的 `weights`。相对路径以仓库根目录为基准。

```yaml
weights:
  sd: weights/v2-1_512-ema-pruned.ckpt
  residual: weights/residual_stage14_gpt0/checkpoints/residual.pt
  stage2: weights/noise/checkpoints/0012000.pt
  abam: weights/abam/不能删除multiscale_pixelmax01/checkpoints/abam.pt
```

### 第二步　运行测试命令

将待处理的 PNG、JPG 或 JPEG 图像放入 `inputs/`，运行：

```bash
ATTN_MODE=sdp python inference.py --input inputs --output results
```

最终 PNG 保存在 `results/abam/images/`，VAE 解码的中间图保存在 `results/stage2/`。可用 `--check` 预查输入和权重路径。默认设备为 `cuda:0`。

默认使用 `a nighttime scene` 作为场景描述。可用 `--prompt` 指定整批场景描述，或用 `--caption-csv` 提供逐图描述。CSV 包含 `image_path` 和 `caption` 两列，按文件名主体匹配输入。

代表性去雾结果可查看 [RealNightHaze](.assets/realnighthaze-example.png) 和 [NHRW](.assets/nhrw-example.png)。

### 可选　准备测试数据集

![测试数据集为 RealNightHaze 和 NHRW。](.assets/datasets.png)

RealNightHaze 可从 GAPSF 的项目页面获取。将测试图像放入同一目录，通过 `--input` 指定。 [RealNightHaze 数据下载入口](https://github.com/jinyeying/nighttime_dehaze#nighttime-haze-data)

## 💪 训练

训练代码待发布。

## 💻 定量评估

下表为论文稿中的无参考图像质量评估结果。↑ 表示越大越好，↓ 表示越小越好。粗体和下划线分别标记最优和次优。

**RealNightHaze**

![RealNightHaze 数据集的八项无参考指标定量对比](.assets/quant-realnighthaze.png)

**NHRW**

![NHRW 数据集的八项无参考指标定量对比](.assets/quant-nhrw.png)

## 📜 许可

代码许可见 [LICENSE](LICENSE)，第三方组件及其声明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 📖 引用

```bibtex
@inproceedings{diffdehaze,
  title     = {基于扩散模型的夜间图像去雾方法},
  author    = {},
  booktitle = {},
  year      = {}
}
```

## 📮 联系方式

**邮箱**
