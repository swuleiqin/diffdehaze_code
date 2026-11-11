import math

import torch
from torch.nn import functional as F

from .histogram import SoftHistogram
from .runtime import checked_prior


def policy_settings(cfg):
    threshold = float(cfg.get("kl_threshold", 0.10))
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("inference.kl_threshold must be finite and nonnegative (nats)")
    if "bottom_fraction" in cfg or "transition_fraction" in cfg:
        raise ValueError("Height-based correction was removed. Delete bottom_fraction/transition_fraction "
                         "and use inference.mathematical_control from the current YAML.")
    raw_control = cfg.get("mathematical_control", {})
    if not isinstance(raw_control, dict):
        raise ValueError("inference.mathematical_control must be a mapping")
    control = {
        "max_gain": float(raw_control.get("max_gain", 2.0)),
        "strength": float(raw_control.get("strength", 1.0)),
        "smoothing_sigma": float(raw_control.get("smoothing_sigma", 5.0)),
        "black_range": raw_control.get("black_range", [0.005, 0.03]),
        "bright_range": raw_control.get("bright_range", [0.10, 0.60]),
    }
    if not math.isfinite(control["max_gain"]) or control["max_gain"] < 1:
        raise ValueError("max_gain must be finite and >=1; 1 means no brightening")
    if not math.isfinite(control["strength"]) or not 0 <= control["strength"] <= 1:
        raise ValueError("strength must be in [0,1]")
    if not math.isfinite(control["smoothing_sigma"]) or not 0 <= control["smoothing_sigma"] <= 64:
        raise ValueError("smoothing_sigma must be in [0,64] pixels; 0 disables smoothing")
    for key in ("black_range", "bright_range"):
        values = control[key]
        if not isinstance(values, list) or len(values) != 2:
            raise ValueError(f"{key} must contain two brightness thresholds")
        values = [float(value) for value in values]
        if not all(math.isfinite(value) for value in values) or not 0 <= values[0] < values[1] <= 1:
            raise ValueError(f"{key} must satisfy 0 <= low < high <= 1")
        control[key] = values
    if control["black_range"][1] > control["bright_range"][0]:
        raise ValueError("black_range high must not exceed bright_range low")
    force = cfg.get("force_process_filenames", [])
    excluded = cfg.get("exclude_dirs", [".ipynb_checkpoints"])
    for values in (force, excluded):
        if not isinstance(values, list) or any(
            not isinstance(name, str) or not name.strip() or name in (".", "..")
            or "/" in name or "\\" in name for name in values
        ):
            raise ValueError("force_process_filenames/exclude_dirs must be lists of plain names")
    force = sorted({name.casefold() for name in force})
    return {"kl_threshold": threshold, "correction_mode": "global_math_v1",
            "mathematical_control": control, "force_process_filenames": force,
            "exclude_dirs": sorted({name.casefold() for name in excluded})}


def screening_reference(checkpoint, device):
    metadata = checkpoint.get("prior_metadata")
    if checkpoint.get("prior") is None or not isinstance(metadata, dict):
        raise ValueError("KL screening requires an embedded clean-night prior. "
                         "A normal-only checkpoint without a prior cannot be screened.")
    size = metadata.get("image_size")
    if type(size) is not int or size < 1:
        raise ValueError("Checkpoint prior metadata is missing a valid image_size")
    histogram = SoftHistogram(**metadata["histogram"]).to(device)
    prior = checked_prior(checkpoint["prior"], histogram.bins).to(device)
    return histogram, prior, size


def illumination_kl(illumination, histogram, prior):
    q = histogram(illumination)
    values = (q * (q.log() - prior.log().unsqueeze(0))).sum(dim=1)
    if not torch.isfinite(values).all():
        raise FloatingPointError("Non-finite input illumination KL")

    return values.clamp_min(0)


def smooth_brightness(illumination, sigma):
    if sigma == 0:
        return illumination
    radius = math.ceil(3 * sigma)
    offsets = torch.arange(-radius, radius + 1, device=illumination.device, dtype=illumination.dtype)
    kernel = torch.exp(-0.5 * (offsets / sigma).square())
    kernel = kernel / kernel.sum()
    horizontal = F.conv2d(F.pad(illumination, (radius, radius, 0, 0), mode="replicate"),
                          kernel.view(1, 1, 1, -1))
    return F.conv2d(F.pad(horizontal, (0, 0, radius, radius), mode="replicate"),
                    kernel.view(1, 1, -1, 1))


def smoothstep(value, low, high):
    position = ((value - low) / (high - low)).clamp(0, 1)
    return position.square() * (3 - 2 * position)


def restrict_correction(image, result, control):
    illumination = result["illumination"]
    brightness = smooth_brightness(illumination, control["smoothing_sigma"])
    visible = smoothstep(brightness, *control["black_range"])
    dark = 1 - smoothstep(brightness, *control["bright_range"])
    weight = (control["strength"] * visible * dark).clamp(0, 1)
    negative_log_l = -illumination.log()
    predicted_log_gain = ((1 - result["gamma"]) * negative_log_l).clamp_min(0)
    log_limit = math.log(control["max_gain"])
    if log_limit == 0:
        final_log_gain = torch.zeros_like(illumination)
    else:

        final_log_gain = weight * log_limit * torch.tanh(predicted_log_gain / log_limit)
        final_log_gain = torch.minimum(final_log_gain, predicted_log_gain)
    gain = final_log_gain.exp()
    effective_gamma = 1 - final_log_gain / negative_log_l.clamp_min(1e-12)
    corrected = illumination * gain
    enhanced = image * gain
    return {"image": enhanced, "gamma": effective_gamma, "illumination": illumination,
            "corrected_illumination": corrected, "correction_weight": weight, "gain": gain}
