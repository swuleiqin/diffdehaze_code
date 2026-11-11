import math
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import torch
from torch import nn

from .attention import BasicTransformerBlock
from .unet import ResBlock


LORA_REGIONS = {
    "encoder": "input_blocks",
    "middle": "middle_block",
    "decoder": "output_blocks",
}


def _cfg_get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(key, default)
    getter = getattr(config, "get", None)
    if getter is not None:
        return getter(key, default)
    return getattr(config, key, default)


def _require_positive(name: str, value: Any) -> float:
    value = float(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


class LoRALinear(nn.Module):
    lora_kind = "attention"

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float,
        scale: float,
        region: str,
        attention_type: str,
        target_name: str,
    ):
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError(f"LoRALinear requires nn.Linear, got {type(base_layer)}")
        rank = int(_require_positive("Attention LoRA rank", rank))
        alpha = _require_positive("Attention LoRA alpha", alpha)
        dropout = float(dropout)
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"Attention LoRA dropout must be in [0, 1), got {dropout}")

        self.base_layer = base_layer
        for parameter in self.base_layer.parameters():
            parameter.requires_grad = False

        self.lora_down = nn.Linear(base_layer.in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, base_layer.out_features, bias=False)
        self.lora_dropout = nn.Dropout(dropout)
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.lora_scale = float(scale)
        self.lora_region = region
        self.lora_attention_type = attention_type
        self.lora_target_name = target_name

        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.base_layer(x)
        update = self.lora_up(self.lora_down(self.lora_dropout(x)))
        return residual + update * (self.lora_scale * self.scaling)

    def lora_parameters(self) -> Iterable[nn.Parameter]:
        yield from self.lora_down.parameters()
        yield from self.lora_up.parameters()


class LoConConv2d(nn.Module):
    lora_kind = "convolution"

    def __init__(
        self,
        base_layer: nn.Conv2d,
        rank: int,
        alpha: float,
        dropout: float,
        scale: float,
        region: str,
    ):
        super().__init__()
        if not isinstance(base_layer, nn.Conv2d):
            raise TypeError(f"LoConConv2d requires nn.Conv2d, got {type(base_layer)}")
        if base_layer.groups != 1:
            raise ValueError(
                "LoConConv2d currently requires groups=1; "
                f"got groups={base_layer.groups}"
            )
        rank = int(_require_positive("Conv LoRA rank", rank))
        alpha = _require_positive("Conv LoRA alpha", alpha)
        dropout = float(dropout)
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"Conv LoRA dropout must be in [0, 1), got {dropout}")

        self.base_layer = base_layer
        for parameter in self.base_layer.parameters():
            parameter.requires_grad = False

        self.lora_down = nn.Conv2d(
            base_layer.in_channels,
            rank,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )
        self.lora_up = nn.Conv2d(
            rank,
            base_layer.out_channels,
            kernel_size=base_layer.kernel_size,
            stride=base_layer.stride,
            padding=base_layer.padding,
            dilation=base_layer.dilation,
            groups=1,
            bias=False,
            padding_mode=base_layer.padding_mode,
        )
        self.lora_dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.lora_scale = float(scale)
        self.lora_region = region

        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.base_layer(x)
        update = self.lora_up(self.lora_down(self.lora_dropout(x)))
        return residual + update * (self.lora_scale * self.scaling)

    def lora_parameters(self) -> Iterable[nn.Parameter]:
        yield from self.lora_down.parameters()
        yield from self.lora_up.parameters()


LORA_MODULE_TYPES = (LoRALinear, LoConConv2d)


def iter_lora_modules(
    module: nn.Module,
    kind: Optional[str] = None,
    region: Optional[str] = None,
) -> Iterable[Tuple[str, nn.Module]]:
    for path, child in module.named_modules():
        if not isinstance(child, LORA_MODULE_TYPES):
            continue
        if kind is not None and child.lora_kind != kind:
            continue
        if region is not None and child.lora_region != region:
            continue
        yield path, child


def _attention_linear_location(
    attention: nn.Module, target_name: str
) -> Tuple[nn.Module, Any, nn.Linear]:
    if target_name != "to_out":
        layer = getattr(attention, target_name, None)
        if not isinstance(layer, nn.Linear):
            raise TypeError(
                f"Expected {attention.__class__.__name__}.{target_name} to be "
                f"nn.Linear, got {type(layer)}"
            )
        return attention, target_name, layer

    to_out = getattr(attention, "to_out", None)
    if not isinstance(to_out, nn.Sequential):
        raise TypeError(
            f"Expected {attention.__class__.__name__}.to_out to be nn.Sequential, "
            f"got {type(to_out)}"
        )
    linear_indices = [
        index for index, child in enumerate(to_out) if isinstance(child, nn.Linear)
    ]
    if linear_indices != [0]:
        raise RuntimeError(
            "Expected exactly to_out[0] to be the sole Linear projection, "
            f"got Linear indices {linear_indices}"
        )
    return to_out, 0, to_out[0]


def _set_child(parent: nn.Module, key: Any, child: nn.Module) -> None:
    if isinstance(parent, nn.Sequential):
        parent[int(key)] = child
    else:
        setattr(parent, str(key), child)


def _inject_attention_lora(
    unet: nn.Module,
    attention_config: Any,
    scale: float,
    report: Dict[str, Any],
) -> None:
    if not bool(_cfg_get(attention_config, "enabled", True)):
        return

    rank = int(_cfg_get(attention_config, "rank", 8))
    alpha = float(_cfg_get(attention_config, "alpha", 16))
    dropout = float(_cfg_get(attention_config, "dropout", 0.05))
    self_config = _cfg_get(attention_config, "self_attention", {})
    cross_config = _cfg_get(attention_config, "cross_attention", {})
    self_targets = list(_cfg_get(self_config, "target_modules", []))
    cross_targets = list(_cfg_get(cross_config, "target_modules", []))

    expected_self = {"to_q", "to_k", "to_v", "to_out"}
    expected_cross = {"to_q", "to_out"}
    if set(self_targets) != expected_self or len(self_targets) != len(expected_self):
        raise ValueError(
            "Self-attention target_modules must contain each of "
            f"{sorted(expected_self)} exactly once; got {self_targets}"
        )
    if set(cross_targets) != expected_cross or len(cross_targets) != len(expected_cross):
        raise ValueError(
            "Cross-attention target_modules must contain each of "
            f"{sorted(expected_cross)} exactly once; got {cross_targets}"
        )

    attention_specs = []
    if bool(_cfg_get(self_config, "enabled", True)):
        attention_specs.append(("attn1", self_targets))
    if bool(_cfg_get(cross_config, "enabled", True)):
        attention_specs.append(("attn2", cross_targets))
    if not attention_specs:
        raise ValueError("Attention LoRA is enabled but attn1 and attn2 are disabled")

    for region, root_name in LORA_REGIONS.items():
        root = getattr(unet, root_name, None)
        if not isinstance(root, nn.Module):
            raise AttributeError(f"UNet has no module named {root_name}")
        transformer_blocks = [
            (path, block)
            for path, block in root.named_modules()
            if isinstance(block, BasicTransformerBlock)
        ]
        if not transformer_blocks:
            raise RuntimeError(
                f"No BasicTransformerBlock found in UNet {region} ({root_name})"
            )

        for block_path, block in transformer_blocks:
            if block.disable_self_attn:
                raise RuntimeError(
                    f"{root_name}.{block_path}.attn1 uses cross-attention because "
                    "disable_self_attn=True; this conflicts with the requested attn1 policy"
                )
            for attention_name, targets in attention_specs:
                attention = getattr(block, attention_name, None)
                if not isinstance(attention, nn.Module):
                    raise TypeError(
                        f"Expected {root_name}.{block_path}.{attention_name} to be a module"
                    )
                for target_name in targets:
                    parent, key, base_layer = _attention_linear_location(
                        attention, target_name
                    )
                    wrapper = LoRALinear(
                        base_layer=base_layer,
                        rank=rank,
                        alpha=alpha,
                        dropout=dropout,
                        scale=scale,
                        region=region,
                        attention_type=attention_name,
                        target_name=target_name,
                    )
                    _set_child(parent, key, wrapper)
                    suffix = "to_out.0" if target_name == "to_out" else target_name
                    full_path = (
                        f"{root_name}.{block_path}.{attention_name}.{suffix}"
                    )
                    report["attention"][region][attention_name][target_name].append(
                        full_path
                    )


def _resblock_second_conv_location(
    block: ResBlock, block_path: str
) -> Tuple[nn.Sequential, int, nn.Conv2d]:
    if block.updown:
        raise RuntimeError(
            f"Refusing Conv LoRA on up/down ResBlock at {block_path}"
        )
    input_convs = [
        child for child in block.in_layers.modules() if isinstance(child, nn.Conv2d)
    ]
    output_convs = [
        (index, child)
        for index, child in enumerate(block.out_layers)
        if isinstance(child, nn.Conv2d)
    ]
    if len(input_convs) != 1 or len(output_convs) != 1:
        raise RuntimeError(
            f"Expected one first conv and one second conv in ResBlock {block_path}; "
            f"found {len(input_convs)} and {len(output_convs)}"
        )
    index, target = output_convs[0]
    if target.kernel_size != (3, 3):
        raise RuntimeError(
            f"ResBlock second conv at {block_path}.out_layers.{index} is not 3x3: "
            f"kernel_size={target.kernel_size}"
        )
    if target is block.skip_connection:
        raise RuntimeError(f"Conv LoRA target at {block_path} is the skip connection")
    return block.out_layers, index, target


def _inject_convolution_lora(
    unet: nn.Module,
    convolution_config: Any,
    scale: float,
    report: Dict[str, Any],
) -> None:
    if not bool(_cfg_get(convolution_config, "enabled", True)):
        return
    if _cfg_get(convolution_config, "target") != "second_resblock_3x3_conv":
        raise ValueError(
            "Conv LoRA target must be 'second_resblock_3x3_conv'"
        )
    if bool(_cfg_get(convolution_config, "apply_to_encoder", False)):
        raise ValueError("Encoder Conv LoRA is forbidden for this configuration")

    rank = int(_cfg_get(convolution_config, "rank", 4))
    alpha = float(_cfg_get(convolution_config, "alpha", 8))
    dropout = float(_cfg_get(convolution_config, "dropout", 0.0))
    enabled_regions = {
        "middle": bool(_cfg_get(convolution_config, "apply_to_middle", True)),
        "decoder": bool(_cfg_get(convolution_config, "apply_to_decoder", True)),
    }
    if not any(enabled_regions.values()):
        raise ValueError("Conv LoRA is enabled but middle and decoder are disabled")

    for region, enabled in enabled_regions.items():
        if not enabled:
            continue
        root_name = LORA_REGIONS[region]
        root = getattr(unet, root_name, None)
        if not isinstance(root, nn.Module):
            raise AttributeError(f"UNet has no module named {root_name}")
        resblocks = [
            (path, block)
            for path, block in root.named_modules()
            if isinstance(block, ResBlock)
        ]
        if not resblocks:
            raise RuntimeError(f"No ResBlock found in UNet {region} ({root_name})")

        for block_path, block in resblocks:
            parent, index, base_layer = _resblock_second_conv_location(
                block, f"{root_name}.{block_path}"
            )
            wrapper = LoConConv2d(
                base_layer=base_layer,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                scale=scale,
                region=region,
            )
            parent[index] = wrapper
            full_path = f"{root_name}.{block_path}.out_layers.{index}"
            report["convolution"][region].append(full_path)


def inject_unet_lora(unet: nn.Module, lora_config: Any) -> Dict[str, Any]:
    existing = list(iter_lora_modules(unet))
    if existing:
        raise RuntimeError(
            f"UNet already contains {len(existing)} LoRA modules; refusing double injection"
        )

    report = {
        "enabled": bool(_cfg_get(lora_config, "enabled", False)),
        "attention": defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        ),
        "convolution": defaultdict(list),
    }
    if not report["enabled"]:
        return report

    scale = float(_cfg_get(lora_config, "scale", 1.0))
    _inject_attention_lora(
        unet,
        _cfg_get(lora_config, "attention", {}),
        scale,
        report,
    )
    _inject_convolution_lora(
        unet,
        _cfg_get(lora_config, "convolution", {}),
        scale,
        report,
    )

    attention_count = sum(
        len(paths)
        for region_data in report["attention"].values()
        for attention_data in region_data.values()
        for paths in attention_data.values()
    )
    convolution_count = sum(
        len(paths) for paths in report["convolution"].values()
    )
    if bool(_cfg_get(_cfg_get(lora_config, "attention", {}), "enabled", True)):
        if attention_count == 0:
            raise RuntimeError("Attention LoRA enabled but no modules were injected")
    if bool(_cfg_get(_cfg_get(lora_config, "convolution", {}), "enabled", True)):
        if convolution_count == 0:
            raise RuntimeError("Conv LoRA enabled but no modules were injected")
    report["attention_count"] = attention_count
    report["convolution_count"] = convolution_count
    report["total_count"] = attention_count + convolution_count
    return report


def set_lora_scale(module: nn.Module, scale: float) -> None:
    scale = float(scale)
    for _, child in iter_lora_modules(module):
        child.lora_scale = scale


def set_lora_train_mode(module: nn.Module, training: bool) -> None:
    for _, child in iter_lora_modules(module):
        child.train(training)


def _expected_lora_tensors(
    unet: nn.Module, kind: str
) -> Dict[str, torch.Tensor]:
    expected = {}
    for path, child in iter_lora_modules(unet, kind=kind):
        expected[f"{path}.lora_down.weight"] = child.lora_down.weight
        expected[f"{path}.lora_up.weight"] = child.lora_up.weight
    return expected


def load_lora_state_dict(
    unet: nn.Module,
    state: Mapping[str, torch.Tensor],
    kind: str,
) -> int:
    expected = _expected_lora_tensors(unet, kind)
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    if missing or unexpected:
        raise RuntimeError(
            f"Strict {kind} LoRA load failed: {len(missing)} missing and "
            f"{len(unexpected)} unexpected keys. Missing examples: {missing[:5]}; "
            f"unexpected examples: {unexpected[:5]}"
        )
    with torch.no_grad():
        for key, target in expected.items():
            source = state[key]
            if tuple(source.shape) != tuple(target.shape):
                raise RuntimeError(
                    f"Shape mismatch for {key}: checkpoint {tuple(source.shape)} "
                    f"vs model {tuple(target.shape)}"
                )
            target.copy_(source.to(device=target.device, dtype=target.dtype))
    return len(expected)


def lora_config_signature(lora_config: Any) -> Dict[str, Any]:
    attention = _cfg_get(lora_config, "attention", {})
    self_attention = _cfg_get(attention, "self_attention", {})
    cross_attention = _cfg_get(attention, "cross_attention", {})
    convolution = _cfg_get(lora_config, "convolution", {})
    return {
        "attention_rank": int(_cfg_get(attention, "rank", 8)),
        "attention_alpha": float(_cfg_get(attention, "alpha", 16)),
        "self_attention_targets": list(
            _cfg_get(self_attention, "target_modules", [])
        ),
        "cross_attention_targets": list(
            _cfg_get(cross_attention, "target_modules", [])
        ),
        "conv_rank": int(_cfg_get(convolution, "rank", 4)),
        "conv_alpha": float(_cfg_get(convolution, "alpha", 8)),
        "conv_encoder": bool(_cfg_get(convolution, "apply_to_encoder", False)),
        "conv_middle": bool(_cfg_get(convolution, "apply_to_middle", True)),
        "conv_decoder": bool(_cfg_get(convolution, "apply_to_decoder", True)),
    }


def validate_lora_config_signature(
    checkpoint_config: Mapping[str, Any], lora_config: Any
) -> None:
    expected = lora_config_signature(lora_config)
    actual = dict(checkpoint_config)
    if actual != expected:
        differences = {
            key: {"checkpoint": actual.get(key), "current": expected.get(key)}
            for key in sorted(set(actual) | set(expected))
            if actual.get(key) != expected.get(key)
        }
        raise RuntimeError(
            f"LoRA checkpoint/config mismatch: {differences}"
        )


def format_injection_report(report: Dict[str, Any]) -> str:
    if not report.get("enabled", False):
        return "LoRA disabled; no modules injected."
    lines = [
        f"Inserted {report['attention_count']} Attention LoRA modules and "
        f"{report['convolution_count']} Conv LoRA modules."
    ]
    for region in ("encoder", "middle", "decoder"):
        for attention_type in ("attn1", "attn2"):
            target_data = report["attention"].get(region, {}).get(
                attention_type, {}
            )
            for target_name in ("to_q", "to_k", "to_v", "to_out"):
                paths = target_data.get(target_name, [])
                if paths:
                    lines.append(
                        f"Attention {region}/{attention_type}/{target_name}: "
                        f"{len(paths)}"
                    )
    for region in ("middle", "decoder"):
        paths = report["convolution"].get(region, [])
        lines.append(f"Conv {region}: {len(paths)}")
        lines.extend(f"  - {path}" for path in paths)
    return "\n".join(lines)


def count_parameters(parameters: Iterable[nn.Parameter]) -> int:
    return sum(parameter.numel() for parameter in parameters)


def assert_lora_scope(control_ldm: nn.Module, lora_enabled: bool) -> None:
    unet_lora = list(iter_lora_modules(control_ldm.unet))
    forbidden = {
        "controlnet": list(iter_lora_modules(control_ldm.controlnet)),
        "vae": list(iter_lora_modules(control_ldm.vae)),
        "clip": list(iter_lora_modules(control_ldm.clip)),
    }
    offenders = {name: paths for name, paths in forbidden.items() if paths}
    if offenders:
        raise RuntimeError(f"LoRA found outside the main UNet: {offenders}")
    if lora_enabled and not unet_lora:
        raise RuntimeError("LoRA is enabled but the main UNet has no LoRA modules")
    if not lora_enabled and unet_lora:
        raise RuntimeError("LoRA is disabled but the main UNet contains LoRA modules")

    lora_parameter_ids = {
        id(parameter)
        for _, child in unet_lora
        for parameter in child.lora_parameters()
    }
    for name, parameter in control_ldm.unet.named_parameters():
        if id(parameter) in lora_parameter_ids:
            if not parameter.requires_grad:
                raise RuntimeError(f"LoRA parameter is frozen: unet.{name}")
        elif parameter.requires_grad:
            raise RuntimeError(f"Base UNet parameter unexpectedly trainable: unet.{name}")
    for module_name in ("vae", "clip"):
        for name, parameter in getattr(control_ldm, module_name).named_parameters():
            if parameter.requires_grad:
                raise RuntimeError(
                    f"{module_name} parameter unexpectedly trainable: {name}"
                )
    for name, parameter in control_ldm.controlnet.named_parameters():
        if not parameter.requires_grad:
            raise RuntimeError(f"ControlNet parameter unexpectedly frozen: {name}")
