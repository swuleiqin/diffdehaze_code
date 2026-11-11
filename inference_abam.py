import json

import torch
from tqdm import tqdm

from abam.data import read_rgb, save_image
from abam.inference_policy import illumination_kl, policy_settings, restrict_correction, screening_reference
from abam.model import AdaptiveBrightnessAdjustmentModule, IlluminationDecomposition
from abam.runtime import load_checkpoint


@torch.no_grad()
def adjust_brightness(cfg, checkpoint_path, restored, output, device, overwrite):
    policy = policy_settings(cfg)
    report_path = output / "screening_report.json"
    checkpoint = load_checkpoint(checkpoint_path)
    histogram, prior, screen_size = screening_reference(checkpoint, device)
    decomposition = IlluminationDecomposition(epsilon=checkpoint["model_config"]["epsilon"]).to(device)
    report = {
        "status": "screening",
        "input_folder": str(restored[0].parent),
        "output_folder": str(output),
        "checkpoint": str(checkpoint_path),
        "policy": policy,
        "kl_direction": "input_illumination || clean_night_prior",
        "kl_units": "nats",
        "screen_image_size": screen_size,
        "histogram": checkpoint["prior_metadata"]["histogram"],
        "images": [],
    }


    for path in tqdm(restored, desc="ABAM KL screening", unit="image"):
        preview = read_rgb(path, image_size=screen_size).unsqueeze(0).to(device)
        _, illumination = decomposition(preview)
        score = float(illumination_kl(illumination, histogram, prior)[0].item())
        forced = path.name.casefold() in policy["force_process_filenames"]
        matches = score <= policy["kl_threshold"]
        report["images"].append({
            "input": path.name,
            "output": "images/" + path.name,
            "kl": score,
            "matches_prior": matches,
            "forced": forced,
            "process": forced or not matches,
        })

    process_count = sum(record["process"] for record in report["images"])
    report["summary"] = {
        "total": len(restored),
        "process": process_count,
        "keep": len(restored) - process_count,
        "forced": sum(record["forced"] for record in report["images"]),
    }
    model = None
    if process_count:
        model = AdaptiveBrightnessAdjustmentModule(**checkpoint["model_config"]).to(device)
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval().requires_grad_(False)
    del checkpoint
    map_names = ("gamma", "illumination", "corrected_illumination", "correction_weight")

    for path, record in tqdm(zip(restored, report["images"]), total=len(restored), desc="ABAM output", unit="image"):
        image = read_rgb(path).unsqueeze(0).to(device)
        if record["process"]:
            result = restrict_correction(image, model(image), policy["mathematical_control"])
        else:
            _, illumination = decomposition(image)
            result = {
                "image": image,
                "gamma": torch.ones_like(illumination),
                "illumination": illumination,
                "corrected_illumination": illumination,
                "correction_weight": torch.zeros_like(illumination),
                "gain": torch.ones_like(illumination),
            }
        if any(not torch.isfinite(value).all() for value in result.values()):
            raise FloatingPointError(f"Non-finite ABAM output: {path}")
        record["applied_gain_mean"] = float(result["gain"].mean().item())
        record["applied_gain_max"] = float(result["gain"].max().item())
        save_image(result["image"][0], output / "images" / path.name)
        if cfg["save_maps"]:
            for name in map_names:
                save_image(result[name][0], output / name / path.name)

    report["status"] = "completed"
    with report_path.open("w" if overwrite else "x", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"Saved {len(restored)} final images to {output / 'images'}.")
