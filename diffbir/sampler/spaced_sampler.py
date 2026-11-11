from __future__ import annotations

from typing import Optional, Tuple, Dict, Literal, List

import torch
import numpy as np
from tqdm import tqdm

from .sampler import Sampler
from ..model.gaussian_diffusion import extract_into_tensor
from ..model.cldm import ControlLDM


def space_timesteps(num_timesteps, section_counts):
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim") :])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)


class SpacedSampler(Sampler):

    def __init__(
        self,
        betas: np.ndarray,
        parameterization: Literal["eps", "v"],
        rescale_cfg: bool,
    ) -> "SpacedSampler":
        super().__init__(betas, parameterization, rescale_cfg)

    def make_schedule(
        self, num_steps: int, start_timestep: int | None = None
    ) -> None:
        num_steps = int(num_steps)
        if start_timestep is None:
            start_timestep = self.num_timesteps - 1
        start_timestep = int(start_timestep)
        if not 0 <= start_timestep < self.num_timesteps:
            raise ValueError(
                f"start_timestep must be in [0, {self.num_timesteps - 1}], "
                f"got {start_timestep}"
            )
        available_timesteps = start_timestep + 1
        minimum_steps = 1 if available_timesteps == 1 else 2
        if not minimum_steps <= num_steps <= available_timesteps:
            raise ValueError(
                f"num_steps must be in [{minimum_steps}, {available_timesteps}] "
                f"when start_timestep={start_timestep}, got {num_steps}"
            )


        used_timesteps = space_timesteps(available_timesteps, str(num_steps))
        betas = []
        last_alpha_cumprod = 1.0
        for i, alpha_cumprod in enumerate(self.training_alphas_cumprod):
            if i > start_timestep:
                break
            if i in used_timesteps:
                betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
        self.timesteps = np.array(
            sorted(list(used_timesteps)), dtype=np.int32
        )
        if self.timesteps[0] != 0 or self.timesteps[-1] != start_timestep:
            raise RuntimeError(
                "Spaced schedule must include both timestep 0 and its configured "
                f"start; got endpoints {self.timesteps[0]} and "
                f"{self.timesteps[-1]} for start_timestep={start_timestep}"
            )

        betas = np.array(betas, dtype=np.float64)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        sqrt_recip_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod)
        sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod - 1)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if len(posterior_variance) == 1:
            posterior_log_variance_clipped = np.log(
                np.maximum(posterior_variance, 1e-20)
            )
        else:
            posterior_log_variance_clipped = np.log(
                np.append(posterior_variance[1], posterior_variance[1:])
            )
        posterior_mean_coef1 = (
            betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        posterior_mean_coef2 = (
            (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)
        )

        self.register("sqrt_alphas_cumprod", np.sqrt(alphas_cumprod))
        self.register("sqrt_one_minus_alphas_cumprod", np.sqrt(1 - alphas_cumprod))
        self.register("sqrt_recip_alphas_cumprod", sqrt_recip_alphas_cumprod)
        self.register("sqrt_recipm1_alphas_cumprod", sqrt_recipm1_alphas_cumprod)
        self.register("posterior_variance", posterior_variance)
        self.register("posterior_log_variance_clipped", posterior_log_variance_clipped)
        self.register("posterior_mean_coef1", posterior_mean_coef1)
        self.register("posterior_mean_coef2", posterior_mean_coef2)

    def q_posterior_mean_variance(
        self, x_start: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor]:
        mean = (
            extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        return mean, variance

    def _predict_xstart_from_eps(
        self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor
    ) -> torch.Tensor:
        return (
            extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_v(
        self, x_t: torch.Tensor, t: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t
            - extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )


    def apply_model(
        self,
        model: ControlLDM,
        x: torch.Tensor,
        model_t: torch.Tensor,
        cond: Dict[str, torch.Tensor],
        uncond: Optional[Dict[str, torch.Tensor]],
        cfg_scale: float,
    ) -> torch.Tensor:
        if uncond is None or cfg_scale == 1.0:
            model_output = model(x, model_t, cond)
        else:
            model_cond = model(x, model_t, cond)
            model_uncond = model(x, model_t, uncond)
            model_output = model_uncond + cfg_scale * (model_cond - model_uncond)
        return model_output

    @torch.no_grad()
    def p_sample(
        self,
        model: ControlLDM,
        x: torch.Tensor,
        model_t: torch.Tensor,
        t: torch.Tensor,
        cond: Dict[str, torch.Tensor],
        uncond: Optional[Dict[str, torch.Tensor]],
        cfg_scale: float,
    ) -> torch.Tensor:

        model_output = self.apply_model(model, x, model_t, cond, uncond, cfg_scale)
        if self.parameterization == "eps":
            pred_x0 = self._predict_xstart_from_eps(x, t, model_output)
        else:
            pred_x0 = self._predict_xstart_from_v(x, t, model_output)


        mean, variance = self.q_posterior_mean_variance(pred_x0, x, t)

        noise = torch.randn_like(x)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        x_prev = mean + nonzero_mask * torch.sqrt(variance) * noise

        return x_prev

    @torch.no_grad()
    def sample(
        self,
        model: ControlLDM,
        device: str,
        steps: int,
        x_size: Tuple[int],
        cond: Dict[str, torch.Tensor],
        uncond: Dict[str, torch.Tensor],
        cfg_scale: float,
        x_T: torch.Tensor | None = None,
        progress: bool = True,
        x_init: torch.Tensor | None = None,
        start_timestep: int | None = None,
        add_initial_noise: bool = True,
    ) -> torch.Tensor:
        self.make_schedule(steps, start_timestep=start_timestep)
        self.to(device)
        if x_init is not None and x_T is not None:
            raise ValueError("Provide only one of x_init and the legacy x_T")
        if x_init is not None:
            if add_initial_noise:
                raise ValueError(
                    "x_init already contains the configured Gaussian noise; "
                    "set add_initial_noise=False to prevent double noising"
                )
            x = x_init
        elif x_T is not None:
            x = x_T
        else:
            if not add_initial_noise:
                raise ValueError(
                    "add_initial_noise=False requires x_init (or legacy x_T)"
                )
            x = torch.randn(x_size, device=device, dtype=torch.float32)

        if tuple(x.shape) != tuple(x_size):
            raise ValueError(
                f"Initial latent shape {tuple(x.shape)} does not match "
                f"x_size={tuple(x_size)}"
            )
        expected_device = torch.device(device)
        if expected_device.type == "cuda" and expected_device.index is None:
            expected_device = torch.device("cuda", torch.cuda.current_device())
        if x.device != expected_device:
            raise ValueError(
                f"Initial latent is on {x.device}, expected {expected_device}"
            )
        timesteps = np.flip(self.timesteps)
        total_steps = len(self.timesteps)
        iterator = tqdm(timesteps, total=total_steps, disable=not progress)
        bs = x.shape[0]

        for i, step in enumerate(iterator):
            model_t = torch.full((bs,), step, device=device, dtype=torch.long)
            t = torch.full((bs,), total_steps - i - 1, device=device, dtype=torch.long)
            cur_cfg_scale = self.get_cfg_scale(cfg_scale, step)
            x = self.p_sample(
                model,
                x,
                model_t,
                t,
                cond,
                uncond,
                cur_cfg_scale,
            )

        return x
