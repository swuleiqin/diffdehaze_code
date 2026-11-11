import cv2
import numpy as np
import torch
from torch.nn.functional import pad
from torchvision.transforms import InterpolationMode, Resize, ToTensor
from torchvision.utils import save_image
from tqdm import tqdm

from diffbir.model import Diffusion, LatentResidualDiffusion
from diffbir.model.latent_residual_diffusion import load_residual_model_checkpoint
from diffbir.model.lora import format_injection_report
from diffbir.sampler import SpacedSampler
from diffbir.utils.common import instantiate_from_config


def load_denoiser(model_config, lora_config, weights, device):
    cldm = instantiate_from_config(model_config)
    sd_checkpoint = torch.load(weights["sd"], map_location="cpu", weights_only=False, mmap=True)
    _, missing = cldm.load_pretrained_sd(sd_checkpoint["state_dict"])
    del sd_checkpoint
    if missing:
        raise RuntimeError(
            f"SD 2.1 Base checkpoint is missing {len(missing)} tensors: {sorted(missing)[:10]}"
        )

    checkpoint = torch.load(weights["stage2"], map_location="cpu", weights_only=True, mmap=True)
    cldm.load_controlnet_from_ckpt(checkpoint)
    print(format_injection_report(cldm.inject_lora(lora_config)))
    cldm.load_lora_from_checkpoint(checkpoint, lora_config)
    cldm.set_lora_scale(lora_config["scale"])
    cldm.eval().requires_grad_(False).to(device)
    return cldm


def load_residual(model_config, settings, diffusion, checkpoint_path, device):
    process = LatentResidualDiffusion(
        residual_model=instantiate_from_config(model_config),
        noise_diffusion=diffusion,
        noise_timestep=settings["noise_timestep"],
        residual_timesteps=settings["timesteps"],
        progress_schedule=settings["progress_schedule"],
        progress_linear_start=settings["progress_linear_start"],
        progress_linear_end=settings["progress_linear_end"],
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
    step = load_residual_model_checkpoint(process, checkpoint, str(checkpoint_path))
    print(f"Loaded residual predictor at step {step}.")
    return process.eval().requires_grad_(False).to(device)


def resize(image, size):
    return Resize(size, interpolation=InterpolationMode.BICUBIC, antialias=True)(image)


def load_image(path, max_side):
    encoded = np.fromfile(str(path), dtype=np.uint8)
    if encoded.size == 0:
        raise ValueError(f"Empty input image: {path}")
    bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not decode input image: {path}")
    image = ToTensor()(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).unsqueeze(0)
    height, width = image.shape[-2:]
    if max(height, width) > max_side:
        scale = max_side / max(height, width)
        image = resize(image, (max(64, round(height * scale)), max(64, round(width * scale))))
    resized_h, resized_w = image.shape[-2:]
    image = pad(image, (0, -resized_w % 64, 0, -resized_h % 64))
    return image, (height, width), (resized_h, resized_w)


@torch.no_grad()
def restore_images(config, weights, jobs, output, device):
    model_config = config["model"]
    inference = config["inference"]
    settings = config["residual"]
    max_side = inference["max_side"]
    sampling_steps = settings["sampling_steps"]
    start = settings["stage2_start_timestep"]
    denoising_steps = inference["denoising_steps"]
    torch.manual_seed(inference["seed"])

    diffusion: Diffusion = instantiate_from_config(model_config["diffusion"])
    if max_side < 64 or not 1 <= sampling_steps < settings["timesteps"]:
        raise ValueError("Check inference.max_side and residual.sampling_steps.")
    if not settings["noise_timestep"] < start < diffusion.num_timesteps:
        raise ValueError("Stage-2 start timestep must be above the fixed residual noise timestep.")
    if not 2 <= denoising_steps <= start + 1:
        raise ValueError("Denoising steps must be between 2 and stage2_start_timestep + 1.")

    cldm = load_denoiser(model_config["cldm"], config["lora"], weights, device)
    residual = load_residual(model_config["residual_model"], settings, diffusion, weights["residual"], device)
    sampler = SpacedSampler(diffusion.betas, diffusion.parameterization, rescale_cfg=False)
    alpha = diffusion.sqrt_alphas_cumprod[start].item()
    sigma = diffusion.sqrt_one_minus_alphas_cumprod[start].item()
    output.mkdir(parents=True, exist_ok=True)
    restored = []

    for path, prompt in tqdm(jobs, desc="Latent restoration", unit="image"):
        image, original_size, (height, width) = load_image(path, max_side)
        cond = cldm.prepare_condition(image.to(device), [prompt])
        z_hazy = cond["c_img"]


        noise = torch.randn_like(z_hazy)
        fixed_noise_state = residual.sample(z_hazy, sampling_steps=sampling_steps, noise=noise)
        clean = (fixed_noise_state - residual.sqrt_one_minus_alpha_bar_h * noise) / residual.sqrt_alpha_bar_h


        initial = alpha * clean + sigma * torch.randn_like(clean)
        latent = sampler.sample(
            model=cldm,
            device=device,
            steps=denoising_steps,
            x_size=z_hazy.shape,
            cond=cond,
            uncond=None,
            cfg_scale=1.0,
            progress=False,
            x_init=initial,
            start_timestep=start,
            add_initial_noise=False,
        )
        decoded = (cldm.vae_decode(latent) + 1.0) / 2.0
        image = resize(decoded[..., :height, :width].clamp(0, 1), original_size)
        target = output / path.with_suffix(".png").name
        save_image(image[0], str(target))
        restored.append(target)

    return restored
