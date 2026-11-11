from typing import Any, Mapping, Tuple, Set, List, Dict

import torch
from torch import nn

from .controlnet import ControlledUnetModel, ControlNet
from .vae import AutoencoderKL
from .util import GroupNorm32
from .clip import FrozenOpenCLIPEmbedder
from .lora import (
    assert_lora_scope,
    inject_unet_lora,
    load_lora_state_dict,
    set_lora_scale,
    set_lora_train_mode,
    validate_lora_config_signature,
)


def disabled_train(self: nn.Module) -> nn.Module:
    return self


class ControlLDM(nn.Module):

    def __init__(
        self, unet_cfg, vae_cfg, clip_cfg, controlnet_cfg, latent_scale_factor
    ):
        super().__init__()
        self.unet = ControlledUnetModel(**unet_cfg)
        self.vae = AutoencoderKL(**vae_cfg)
        self.clip = FrozenOpenCLIPEmbedder(**clip_cfg)
        self.controlnet = ControlNet(**controlnet_cfg)
        self.scale_factor = latent_scale_factor
        self.control_scales = [1.0] * 13
        self.lora_report = None

    @torch.no_grad()
    def load_pretrained_sd(
        self, sd: Dict[str, torch.Tensor]
    ) -> Tuple[Set[str], Set[str]]:
        module_map = {
            "unet": "model.diffusion_model",
            "vae": "first_stage_model",
            "clip": "cond_stage_model",
        }
        modules = [("unet", self.unet), ("vae", self.vae), ("clip", self.clip)]
        used = set()
        missing = set()
        for name, module in modules:
            init_sd = {}
            scratch_sd = module.state_dict()
            for key in scratch_sd:
                target_key = ".".join([module_map[name], key])
                if target_key not in sd:
                    missing.add(target_key)
                    continue
                init_sd[key] = sd[target_key]
                used.add(target_key)
            module.load_state_dict(init_sd, strict=False)
        unused = set(sd.keys()) - used
        for module in [self.vae, self.clip, self.unet]:
            module.eval()
            module.train = disabled_train
            for p in module.parameters():
                p.requires_grad = False
        return unused, missing

    @torch.no_grad()
    def load_controlnet_from_ckpt(self, checkpoint: Mapping[str, Any]) -> str:
        if not isinstance(checkpoint, Mapping):
            raise TypeError(
                f"ControlNet checkpoint must be a mapping, got {type(checkpoint)}"
            )
        if "controlnet" in checkpoint:
            controlnet_state = checkpoint["controlnet"]
            checkpoint_format = "combined"
        else:
            if not checkpoint or not all(
                isinstance(value, torch.Tensor) for value in checkpoint.values()
            ):
                raise RuntimeError(
                    "Checkpoint is neither a combined Stage-2 checkpoint nor a "
                    "legacy pure ControlNet state_dict"
                )
            controlnet_state = checkpoint
            checkpoint_format = "legacy_controlnet"
        self.controlnet.load_state_dict(controlnet_state, strict=True)
        return checkpoint_format

    def inject_lora(self, lora_config: Any) -> Dict[str, Any]:
        self.lora_report = inject_unet_lora(self.unet, lora_config)
        lora_enabled = bool(
            lora_config.get("enabled", False)
            if hasattr(lora_config, "get")
            else getattr(lora_config, "enabled", False)
        )
        assert_lora_scope(
            self,
            lora_enabled=lora_enabled,
        )
        return self.lora_report

    def set_lora_scale(self, scale: float) -> None:
        set_lora_scale(self.unet, scale)

    def set_lora_train_mode(self, training: bool) -> None:
        set_lora_train_mode(self.unet, training)


    def load_lora_from_checkpoint(
        self,
        checkpoint: Mapping[str, Any],
        lora_config: Any,
    ) -> Dict[str, int]:
        required = {
            "attention_lora",
            "conv_lora",
            "lora_config",
        }
        missing_sections = sorted(required - set(checkpoint))
        if missing_sections:
            raise RuntimeError(
                f"LoRA checkpoint is missing sections: {missing_sections}"
            )
        validate_lora_config_signature(checkpoint["lora_config"], lora_config)
        return {
            "attention_tensors": load_lora_state_dict(
                self.unet, checkpoint["attention_lora"], "attention"
            ),
            "conv_tensors": load_lora_state_dict(
                self.unet, checkpoint["conv_lora"], "convolution"
            ),
        }


    def vae_encode(self, image: torch.Tensor) -> torch.Tensor:
        return self.vae.encode(image).mode() * self.scale_factor

    def vae_decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(z / self.scale_factor)

    def prepare_condition(
        self, cond_img: torch.Tensor, txt: List[str]
    ) -> Dict[str, torch.Tensor]:
        return dict(
            c_txt=self.clip.encode(txt),
            c_img=self.vae_encode(cond_img * 2 - 1),
        )

    def forward(self, x_noisy, t, cond):
        c_txt = cond["c_txt"]
        c_img = cond["c_img"]
        control = self.controlnet(x=x_noisy, hint=c_img, timesteps=t, context=c_txt)
        control = [c * scale for c, scale in zip(control, self.control_scales)]
        eps = self.unet(
            x=x_noisy,
            timesteps=t,
            context=c_txt,
            control=control,
            only_mid_control=False,
        )
        return eps

    def cast_dtype(self, dtype: torch.dtype) -> "ControlLDM":
        self.unet.dtype = dtype
        self.controlnet.dtype = dtype

        for module in [
            self.unet.input_blocks,
            self.unet.middle_block,
            self.unet.output_blocks,
        ]:
            module.type(dtype)

        for module in [
            self.controlnet.input_blocks,
            self.controlnet.zero_convs,
            self.controlnet.middle_block,
            self.controlnet.middle_block_out,
        ]:
            module.type(dtype)

        def cast_groupnorm_32(m):
            if isinstance(m, GroupNorm32):
                m.type(torch.float32)


        for module in [
            self.unet.input_blocks,
            self.unet.middle_block,
            self.unet.output_blocks,
        ]:
            module.apply(cast_groupnorm_32)
        for module in [
            self.controlnet.input_blocks,
            self.controlnet.zero_convs,
            self.controlnet.middle_block,
            self.controlnet.middle_block_out,
        ]:
            module.apply(cast_groupnorm_32)
