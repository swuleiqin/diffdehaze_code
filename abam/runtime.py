import torch

from .model import ILLUMINATION_METHOD


FORMAT = "abam_brighten_only_pixelmax_v2"


def check_model_config(model_config):
    if "illumination_kernel" in model_config:
        raise ValueError("Old illumination_kernel was removed. Use illumination_method: "
                         "pixel_rgb_max_v1, a new prior/exp_dir, and retrain ABAM.")
    if model_config.get("illumination_method") != ILLUMINATION_METHOD:
        raise ValueError("ABAM model_config must explicitly specify illumination_method: pixel_rgb_max_v1")


def checked_prior(tensor, bins):
    if not isinstance(tensor, torch.Tensor) or tensor.shape != (bins,):
        raise ValueError("Invalid cached prior shape")
    if not torch.isfinite(tensor).all() or not (tensor > 0).all():
        raise ValueError("Prior must have finite positive probabilities")
    return tensor.float() / tensor.float().sum()


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") == "abam_brighten_only_v1":
        raise ValueError("Old ABAM checkpoint uses a different illumination decomposition. "
                         "Retrain with pixel_rgb_max_v1; do not resume, initialize or infer with old weights.")
    if checkpoint.get("format") != FORMAT:
        raise ValueError("Not a pixel-max ABAM checkpoint; SD/ControlNet/residual weights cannot be used here")
    check_model_config(checkpoint["model_config"])
    if checkpoint.get("prior") is not None:
        expected = {"method": ILLUMINATION_METHOD, "epsilon": checkpoint["model_config"]["epsilon"]}
        if (checkpoint.get("prior_metadata") or {}).get("decomposition") != expected:
            raise ValueError("Checkpoint prior decomposition does not match pixel-max model")
    return checkpoint
