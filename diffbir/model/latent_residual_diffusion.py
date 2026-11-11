from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from .gaussian_diffusion import Diffusion
from .residual_unet import LatentResidualUNet


RESIDUAL_CHECKPOINT_VERSION = 1


class LatentResidualDiffusion(nn.Module):

    def __init__(
        self,
        residual_model: LatentResidualUNet,
        noise_diffusion: Diffusion,
        noise_timestep: int = 354,
        residual_timesteps: int = 1000,
        progress_schedule: str = "drdd",
        progress_linear_start: float = 1e-4,
        progress_linear_end: float = 2e-2,
    ):
        super().__init__()
        if noise_diffusion.parameterization != "eps":
            raise ValueError(
                "The second-stage model must use eps parameterization so that "
                "the residual output can be handed to its noise predictor"
            )
        if not 0 <= int(noise_timestep) < noise_diffusion.num_timesteps:
            raise ValueError(
                f"noise_timestep must be in [0, {noise_diffusion.num_timesteps - 1}], "
                f"got {noise_timestep}"
            )
        if int(residual_timesteps) < 2:
            raise ValueError("residual_timesteps must be at least 2")
        if residual_model.latent_channels != residual_model.condition_channels:
            raise ValueError(
                "This paired latent process requires equal state and condition "
                "channel counts"
            )
        if residual_model.out_channels != residual_model.latent_channels:
            raise ValueError(
                "Residual-model output channels must equal its state channels"
            )

        self.residual_model = residual_model
        self.noise_timestep = int(noise_timestep)
        self.residual_timesteps = int(residual_timesteps)
        self.noise_num_timesteps = int(noise_diffusion.num_timesteps)
        self.progress_schedule = str(progress_schedule).lower()
        self.progress_linear_start = float(progress_linear_start)
        self.progress_linear_end = float(progress_linear_end)
        if self.progress_schedule not in {"drdd", "linear"}:
            raise ValueError(
                "progress_schedule must be either 'drdd' or 'linear'"
            )
        if not 0 < self.progress_linear_start < self.progress_linear_end < 1:
            raise ValueError(
                "Residual progress beta endpoints must satisfy "
                "0 < start < end < 1"
            )
        self.register_buffer(
            "sqrt_alpha_bar_h",
            noise_diffusion.sqrt_alphas_cumprod[self.noise_timestep]
            .detach()
            .clone()
            .float(),
        )
        self.register_buffer(
            "sqrt_one_minus_alpha_bar_h",
            noise_diffusion.sqrt_one_minus_alphas_cumprod[self.noise_timestep]
            .detach()
            .clone()
            .float(),
        )
        if self.progress_schedule == "linear":
            residual_progress = torch.linspace(
                0.0, 1.0, self.residual_timesteps, dtype=torch.float32
            )
        else:


            progress_betas = torch.linspace(
                self.progress_linear_start,
                self.progress_linear_end,
                self.residual_timesteps,
                dtype=torch.float32,
            )
            progress_alpha_bar = torch.cumprod(1.0 - progress_betas, dim=0)
            residual_progress = 1.0 - torch.sqrt(progress_alpha_bar)
            residual_progress = (
                residual_progress - residual_progress[0]
            ) / (residual_progress[-1] - residual_progress[0])
            residual_progress[0] = 0.0
            residual_progress[-1] = 1.0
        self.register_buffer("residual_progress", residual_progress)


    def _model_time(self, residual_timestep: torch.Tensor) -> torch.Tensor:
        progress = self.residual_progress.gather(0, residual_timestep)
        return progress * float(self.residual_timesteps)


    def make_sampling_indices(
        self, sampling_steps: int, device: torch.device
    ) -> torch.Tensor:
        sampling_steps = int(sampling_steps)
        if sampling_steps < 1:
            raise ValueError("sampling_steps must be positive")
        if sampling_steps > self.residual_timesteps - 1:
            raise ValueError(
                f"sampling_steps cannot exceed {self.residual_timesteps - 1}"
            )
        indices = torch.linspace(
            self.residual_timesteps - 1,
            0,
            sampling_steps + 1,
            device=device,
            dtype=torch.float64,
        ).round().long()
        if not torch.all(indices[:-1] > indices[1:]):
            raise RuntimeError("Residual sampling indices must be strictly decreasing")
        return indices

    @torch.no_grad()
    def sample(
        self,
        z_hazy: torch.Tensor,
        sampling_steps: int,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if z_hazy.ndim != 4:
            raise ValueError("z_hazy must be a BCHW tensor")
        if z_hazy.shape[1] != self.residual_model.condition_channels:
            raise ValueError(
                f"Expected {self.residual_model.condition_channels} hazy latent "
                f"channels, got {z_hazy.shape[1]}"
            )
        if noise is None:
            noise = torch.randn_like(z_hazy)
        elif (
            noise.shape != z_hazy.shape
            or noise.device != z_hazy.device
            or noise.dtype != z_hazy.dtype
        ):
            raise ValueError(
                "noise must match the shape, device and dtype of z_hazy"
            )


        state = (
            self.sqrt_alpha_bar_h * z_hazy
            + self.sqrt_one_minus_alpha_bar_h * noise
        )
        indices = self.make_sampling_indices(sampling_steps, z_hazy.device)
        batch_size = z_hazy.shape[0]
        for current_index, next_index in zip(indices[:-1], indices[1:]):
            current = int(current_index.item())
            following = int(next_index.item())
            residual_timestep = torch.full(
                (batch_size,),
                current,
                device=z_hazy.device,
                dtype=torch.long,
            )
            prediction = self.residual_model(
                state,
                z_hazy,
                self._model_time(residual_timestep),
            )
            progress_delta = (
                self.residual_progress[current]
                - self.residual_progress[following]
            )
            state = state - progress_delta * prediction
        return state


def load_residual_model_checkpoint(
    process: LatentResidualDiffusion,
    checkpoint: Mapping,
    checkpoint_path: str = "<in-memory checkpoint>",
) -> int:

    def checkpoint_float(key: str) -> float:
        value = checkpoint[key]
        if not isinstance(value, (int, float)):
            raise TypeError(
                f"Residual checkpoint {key} must be numeric, got "
                f"{type(value)} in {checkpoint_path}"
            )
        return float(value)

    if not isinstance(checkpoint, Mapping):
        raise TypeError(
            f"Residual checkpoint must be a mapping, got {type(checkpoint)}"
        )
    required = {
        "format_version",
        "residual_model",
        "noise_timestep",
        "noise_num_timesteps",
        "residual_timesteps",
        "progress_schedule",
        "progress_linear_start",
        "progress_linear_end",
        "sqrt_alpha_bar_h",
        "sqrt_one_minus_alpha_bar_h",
        "residual_model_signature",
        "global_step",
    }
    missing = sorted(required - set(checkpoint))
    if missing:
        raise RuntimeError(
            f"Residual checkpoint is missing sections {missing}: {checkpoint_path}"
        )
    if int(checkpoint["format_version"]) != RESIDUAL_CHECKPOINT_VERSION:
        raise RuntimeError(
            f"Unsupported residual checkpoint format "
            f"{checkpoint['format_version']}: {checkpoint_path}"
        )
    expected_metadata = {
        "noise_timestep": process.noise_timestep,
        "noise_num_timesteps": process.noise_num_timesteps,
        "residual_timesteps": process.residual_timesteps,
    }
    for key, expected_integer in expected_metadata.items():
        actual_integer = int(checkpoint[key])
        if actual_integer != expected_integer:
            raise RuntimeError(
                f"Residual checkpoint {key} mismatch: expected "
                f"{expected_integer}, got {actual_integer} in {checkpoint_path}"
            )
    expected_progress_values: dict[str, float] = {
        "progress_linear_start": process.progress_linear_start,
        "progress_linear_end": process.progress_linear_end,
    }
    if str(checkpoint["progress_schedule"]) != process.progress_schedule:
        raise RuntimeError(
            "Residual checkpoint progress_schedule mismatch: expected "
            f"{process.progress_schedule}, got {checkpoint['progress_schedule']} "
            f"in {checkpoint_path}"
        )
    for key in ("progress_linear_start", "progress_linear_end"):
        expected_progress_value = expected_progress_values[key]
        actual_progress_value = checkpoint_float(key)
        if abs(actual_progress_value - expected_progress_value) > 1e-12:
            raise RuntimeError(
                f"Residual checkpoint {key} mismatch: expected "
                f"{expected_progress_value}, got {actual_progress_value} in "
                f"{checkpoint_path}"
            )
    expected_coefficients = {
        "sqrt_alpha_bar_h": float(process.sqrt_alpha_bar_h.item()),
        "sqrt_one_minus_alpha_bar_h": float(
            process.sqrt_one_minus_alpha_bar_h.item()
        ),
    }
    for key, expected_coefficient in expected_coefficients.items():
        actual_coefficient = checkpoint_float(key)
        if abs(actual_coefficient - expected_coefficient) > 1e-7:
            raise RuntimeError(
                f"Residual checkpoint {key} mismatch: expected "
                f"{expected_coefficient:.9f}, got {actual_coefficient:.9f} in "
                f"{checkpoint_path}. The Stage-2 Gaussian schedule must be "
                "identical during training and inference."
            )
    expected_signature = process.residual_model.config_signature
    actual_signature = checkpoint["residual_model_signature"]
    if actual_signature != expected_signature:
        raise RuntimeError(
            "Residual-model architecture mismatch between checkpoint and current "
            f"config: expected {expected_signature}, got {actual_signature} in "
            f"{checkpoint_path}"
        )
    process.residual_model.load_state_dict(
        checkpoint["residual_model"], strict=True
    )
    return int(checkpoint["global_step"])
