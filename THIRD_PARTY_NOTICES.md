# Third-party notices

The model components in `diffbir/` derive from [DiffBIR](https://github.com/XPixelGroup/DiffBIR).
The repository includes its existing Apache License 2.0 in [LICENSE](LICENSE).

`diffbir/model/open_clip/model.py` was adapted from OpenAI CLIP.
`diffbir/model/open_clip/tokenizer.py` was copied from OpenAI CLIP.
The tokenizer uses the accompanying BPE vocabulary.

## OpenAI CLIP

MIT License

Copyright (c) 2021 OpenAI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Source references retained from the model files

- `diffbir/model/attention.py` and `diffbir/model/vae.py` reference [Diffusers attention](https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223).
- `diffbir/model/distributions.py`, `diffbir/model/util.py` and `diffbir/sampler/spaced_sampler.py` reference [guided-diffusion](https://github.com/openai/guided-diffusion).
- `diffbir/model/util.py` also references [improved-diffusion](https://github.com/openai/improved-diffusion) and [denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch).
- `diffbir/model/unet.py` references [the original DDPM implementation](https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66).
- `diffbir/model/gaussian_diffusion.py` references [zero terminal SNR](https://github.com/Max-We/sf-zero-signal-to-noise/blob/main/common_diffusion_noise_schedulers_are_flawed.ipynb).

The separately downloaded Stable Diffusion 2.1 Base weights are distributed under the terms listed in their [model repository](https://huggingface.co/sd2-community/stable-diffusion-2-1-base).
