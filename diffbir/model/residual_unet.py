from __future__ import annotations

from functools import partial
import math
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F


class Residual(nn.Module):
    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(x) + x


def _upsample(dim: int, dim_out: int) -> nn.Module:
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(dim, dim_out, 3, padding=1),
    )


def _downsample(dim: int, dim_out: int) -> nn.Module:
    return nn.Conv2d(dim, dim_out, 4, stride=2, padding=1)


class WeightStandardizedConv2d(nn.Conv2d):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        weight = self.weight
        mean = weight.mean(dim=(1, 2, 3), keepdim=True)
        variance = weight.var(dim=(1, 2, 3), unbiased=False, keepdim=True)
        weight = (weight - mean) * torch.rsqrt(variance + eps)
        return F.conv2d(
            x,
            weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class LayerNorm2d(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        variance, mean = torch.var_mean(
            x, dim=1, unbiased=False, keepdim=True
        )
        return (x - mean) * torch.rsqrt(variance + eps) * self.gain


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.norm = LayerNorm2d(dim)
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(self.norm(x))


class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        if dim < 4 or dim % 2 != 0:
            raise ValueError("Sinusoidal embedding dimension must be even and >= 4")
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        exponent = -math.log(10000.0) / (half_dim - 1)
        frequencies = torch.exp(
            torch.arange(half_dim, device=time.device, dtype=torch.float32)
            * exponent
        )
        angles = time.float()[:, None] * frequencies[None, :]
        return torch.cat((angles.sin(), angles.cos()), dim=-1)


class Block(nn.Module):
    def __init__(self, dim: int, dim_out: int, groups: int):
        super().__init__()
        if dim_out % groups != 0:
            raise ValueError(
                f"GroupNorm groups ({groups}) must divide channels ({dim_out})"
            )
        self.projection = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.activation = nn.SiLU()

    def forward(
        self,
        x: torch.Tensor,
        scale_shift: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        x = self.projection(x)
        x = self.norm(x)
        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        return self.activation(x)


class ResnetBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        *,
        time_embedding_dim: int,
        groups: int,
    ):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embedding_dim, dim_out * 2),
        )
        self.block1 = Block(dim, dim_out, groups)
        self.block2 = Block(dim_out, dim_out, groups)
        self.residual_projection = (
            nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()
        )

    def forward(
        self, x: torch.Tensor, time_embedding: torch.Tensor
    ) -> torch.Tensor:
        scale_shift = self.time_mlp(time_embedding)
        scale_shift = scale_shift[:, :, None, None].chunk(2, dim=1)
        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.residual_projection(x)


class LinearAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head**-0.5
        hidden_dim = heads * dim_head
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            LayerNorm2d(dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = x.shape
        q, k, v = self.to_qkv(x).chunk(3, dim=1)

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.reshape(
                batch, self.heads, self.dim_head, height * width
            )

        q, k, v = map(split_heads, (q, k, v))
        q = q.softmax(dim=-2) * self.scale
        k = k.softmax(dim=-1)
        v = v / (height * width)
        context = torch.einsum("bhdn,bhen->bhde", k, v)
        out = torch.einsum("bhde,bhdn->bhen", context, q)
        out = out.reshape(batch, self.heads * self.dim_head, height, width)
        return self.to_out(out)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head**-0.5
        hidden_dim = heads * dim_head
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = x.shape
        q, k, v = self.to_qkv(x).chunk(3, dim=1)

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.reshape(
                batch, self.heads, self.dim_head, height * width
            )

        q, k, v = map(split_heads, (q, k, v))
        q = q * self.scale
        similarity = torch.einsum("bhdi,bhdj->bhij", q, k)
        attention = similarity.softmax(dim=-1)
        out = torch.einsum("bhij,bhdj->bhid", attention, v)
        out = out.permute(0, 1, 3, 2).reshape(
            batch, self.heads * self.dim_head, height, width
        )
        return self.to_out(out)


class LatentResidualUNet(nn.Module):

    def __init__(
        self,
        dim: int = 64,
        init_dim: int | None = None,
        dim_mults: Sequence[int] = (1, 2, 4, 8),
        latent_channels: int = 4,
        condition_channels: int = 4,
        out_channels: int = 4,
        resnet_block_groups: int = 8,
        attention_heads: int = 4,
        attention_dim_head: int = 32,
    ):
        super().__init__()
        dim = int(dim)
        dim_mults = tuple(int(multiplier) for multiplier in dim_mults)
        resnet_block_groups = int(resnet_block_groups)
        attention_heads = int(attention_heads)
        attention_dim_head = int(attention_dim_head)
        if len(dim_mults) < 2:
            raise ValueError("dim_mults must contain at least two resolutions")
        if any(multiplier <= 0 for multiplier in dim_mults):
            raise ValueError("All dim_mults values must be positive")
        if dim <= 0 or resnet_block_groups <= 0:
            raise ValueError("dim and resnet_block_groups must be positive")
        if latent_channels <= 0 or condition_channels <= 0 or out_channels <= 0:
            raise ValueError("Input and output channel counts must be positive")
        if attention_heads <= 0 or attention_dim_head <= 0:
            raise ValueError("Attention head counts and dimensions must be positive")

        self.latent_channels = int(latent_channels)
        self.condition_channels = int(condition_channels)
        self.out_channels = int(out_channels)
        init_dim = int(init_dim if init_dim is not None else dim)
        if init_dim <= 0:
            raise ValueError("init_dim must be positive")
        self.config_signature = {
            "dim": dim,
            "init_dim": init_dim,
            "dim_mults": list(dim_mults),
            "latent_channels": self.latent_channels,
            "condition_channels": self.condition_channels,
            "out_channels": self.out_channels,
            "resnet_block_groups": resnet_block_groups,
            "attention_heads": attention_heads,
            "attention_dim_head": attention_dim_head,
        }
        input_channels = self.latent_channels + self.condition_channels
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding=3)

        dims = [init_dim, *[int(dim * multiplier) for multiplier in dim_mults]]
        in_out = list(zip(dims[:-1], dims[1:]))
        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbedding(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )
        block = partial(
            ResnetBlock,
            time_embedding_dim=time_dim,
            groups=resnet_block_groups,
        )

        self.downs = nn.ModuleList()
        for index, (dim_in, dim_out) in enumerate(in_out):
            is_last = index == len(in_out) - 1
            self.downs.append(
                nn.ModuleList(
                    [
                        block(dim_in, dim_in),
                        block(dim_in, dim_in),
                        Residual(
                            PreNorm(
                                dim_in,
                                LinearAttention(
                                    dim_in,
                                    heads=attention_heads,
                                    dim_head=attention_dim_head,
                                ),
                            )
                        ),
                        (
                            nn.Conv2d(dim_in, dim_out, 3, padding=1)
                            if is_last
                            else _downsample(dim_in, dim_out)
                        ),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block(mid_dim, mid_dim)
        self.mid_attention = Residual(
            PreNorm(
                mid_dim,
                Attention(
                    mid_dim,
                    heads=attention_heads,
                    dim_head=attention_dim_head,
                ),
            )
        )
        self.mid_block2 = block(mid_dim, mid_dim)

        self.ups = nn.ModuleList()
        for index, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = index == len(in_out) - 1
            self.ups.append(
                nn.ModuleList(
                    [
                        block(dim_out + dim_in, dim_out),
                        block(dim_out + dim_in, dim_out),
                        Residual(
                            PreNorm(
                                dim_out,
                                LinearAttention(
                                    dim_out,
                                    heads=attention_heads,
                                    dim_head=attention_dim_head,
                                ),
                            )
                        ),
                        (
                            nn.Conv2d(dim_out, dim_in, 3, padding=1)
                            if is_last
                            else _upsample(dim_out, dim_in)
                        ),
                    ]
                )
            )

        self.final_res_block = block(init_dim * 2, dim)
        self.final_conv = nn.Conv2d(dim, self.out_channels, 1)
        self.required_multiple = 2 ** (len(in_out) - 1)

    def _validate_inputs(
        self,
        state: torch.Tensor,
        condition: torch.Tensor,
        residual_time: torch.Tensor,
    ) -> None:
        if state.ndim != 4 or condition.ndim != 4:
            raise ValueError("state and condition must be BCHW tensors")
        if (
            state.shape[0] != condition.shape[0]
            or state.shape[2:] != condition.shape[2:]
        ):
            raise ValueError(
                "state and condition must have identical batch and spatial dimensions"
            )
        if state.shape[1] != self.latent_channels:
            raise ValueError(
                f"Expected {self.latent_channels} state channels, got {state.shape[1]}"
            )
        if condition.shape[1] != self.condition_channels:
            raise ValueError(
                f"Expected {self.condition_channels} condition channels, "
                f"got {condition.shape[1]}"
            )
        if residual_time.ndim != 1 or residual_time.shape[0] != state.shape[0]:
            raise ValueError("residual_time must have shape [batch]")

    def _pad_to_valid_size(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        height, width = x.shape[-2:]
        pad_height = (
            self.required_multiple - height % self.required_multiple
        ) % self.required_multiple
        pad_width = (
            self.required_multiple - width % self.required_multiple
        ) % self.required_multiple
        if pad_height == 0 and pad_width == 0:
            return x, height, width
        if height <= pad_height or width <= pad_width:
            raise ValueError(
                "Latent spatial dimensions are too small for reflection padding: "
                f"shape={height}x{width}, padding={pad_height}x{pad_width}"
            )
        return (
            F.pad(x, (0, pad_width, 0, pad_height), mode="reflect"),
            height,
            width,
        )

    def forward(
        self,
        state: torch.Tensor,
        condition: torch.Tensor,
        residual_time: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_inputs(state, condition, residual_time)
        x = torch.cat((state, condition), dim=1)
        x, original_height, original_width = self._pad_to_valid_size(x)
        x = self.init_conv(x)
        initial_features = x
        time_embedding = self.time_mlp(residual_time)
        skips = []

        for block1, block2, attention, downsample in self.downs:
            x = block1(x, time_embedding)
            skips.append(x)
            x = block2(x, time_embedding)
            x = attention(x)
            skips.append(x)
            x = downsample(x)

        x = self.mid_block1(x, time_embedding)
        x = self.mid_attention(x)
        x = self.mid_block2(x, time_embedding)

        for block1, block2, attention, upsample in self.ups:
            x = torch.cat((x, skips.pop()), dim=1)
            x = block1(x, time_embedding)
            x = torch.cat((x, skips.pop()), dim=1)
            x = block2(x, time_embedding)
            x = attention(x)
            x = upsample(x)

        if skips:
            raise RuntimeError("Residual U-Net skip stack was not fully consumed")
        x = torch.cat((x, initial_features), dim=1)
        x = self.final_res_block(x, time_embedding)
        x = self.final_conv(x)
        return x[..., :original_height, :original_width].contiguous()
