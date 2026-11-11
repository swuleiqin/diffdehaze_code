import argparse
import csv
from pathlib import Path, PureWindowsPath
import unicodedata

import yaml


ROOT = Path(__file__).resolve().parent
SUFFIXES = {".jpg", ".jpeg", ".png"}


def project_path(value):
    return (ROOT / Path(value).expanduser()).resolve()


def stem_key(name):
    return unicodedata.normalize("NFC", Path(name).stem).casefold()


def read_captions(path):
    content = path.read_bytes()
    try:
        decoded = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        decoded = content.decode("gb18030")

    reader = csv.DictReader(decoded.splitlines(keepends=True))
    if not {"image_path", "caption"}.issubset(reader.fieldnames or ()):
        raise ValueError(f"Expected image_path and caption columns: {path}")

    captions = {}
    for row in reader:
        name = (row["image_path"] or "").strip()
        caption = (row["caption"] or "").strip()
        key = stem_key(PureWindowsPath(name).name)
        if not key or not caption or key in captions:
            raise ValueError(f"Invalid or duplicate caption at row {reader.line_num}: {path}")
        captions[key] = caption
    return captions


def plan_run(config, args):
    inference = config["inference"]
    weights = {}
    problems = []
    for name in ("sd", "residual", "stage2", "abam"):
        value = config["weights"].get(name)
        if not value:
            problems.append(f"Set weights.{name} in {args.config}")
            continue
        weights[name] = project_path(value)
        if not weights[name].is_file():
            problems.append(f"Missing weights.{name}: {weights[name]}")

    source = project_path(args.input or inference["image_folder"])
    if not source.is_dir():
        problems.append(f"Input folder does not exist: {source}")
    if problems:
        raise ValueError("\n".join(problems))

    images = sorted(
        path for path in source.iterdir() if path.is_file() and path.suffix.lower() in SUFFIXES
    )
    if not images:
        raise ValueError(f"No JPG, JPEG or PNG inputs in {source}")
    keys = [stem_key(path.name) for path in images]
    if len(keys) != len(set(keys)):
        raise ValueError(f"Input filenames must have unique stems: {source}")

    if args.output:
        output_root = project_path(args.output)
        stage2_output = (output_root / "stage2").resolve()
        abam_output = (output_root / "abam").resolve()
    else:
        stage2_output = project_path(inference["result_folder"])
        abam_output = project_path(config["abam"]["output_folder"])
    folders = (source, stage2_output, abam_output)
    for index, left in enumerate(folders):
        for right in folders[index + 1:]:
            if left == right or left in right.parents or right in left.parents:
                raise ValueError(f"Input and output folders must be separate: {left}, {right}")
    for folder in (stage2_output, abam_output):
        if folder.exists() and not folder.is_dir():
            raise ValueError(f"Output path is not a directory: {folder}")

    overwrite = args.overwrite or config["abam"].get("overwrite", False)
    abam_targets = [abam_output / "screening_report.json"]
    output_names = [path.with_suffix(".png").name for path in images]
    abam_targets.extend(abam_output / "images" / name for name in output_names)
    if config["abam"]["save_maps"]:
        for folder in ("gamma", "illumination", "corrected_illumination", "correction_weight"):
            abam_targets.extend(abam_output / folder / name for name in output_names)
    if not overwrite:
        existing = next((path for path in abam_targets if path.exists()), None)
        if existing is not None:
            raise FileExistsError(f"Output already exists: {existing}. Use --output or --overwrite.")

    caption_csv = None if args.prompt is not None else args.caption_csv or inference.get("caption_csv")
    if caption_csv:
        captions = read_captions(project_path(caption_csv))
        missing = [path.name for path, key in zip(images, keys) if key not in captions]
        if missing:
            raise ValueError(f"Missing captions for {len(missing)} inputs: {', '.join(missing[:10])}")
        base_prompts = [captions[key] for key in keys]
    else:
        prompt = args.prompt if args.prompt is not None else inference["prompt"]
        base_prompts = [prompt] * len(images)
    suffix = inference["prompt_suffix"].strip().strip(",")
    prompts = [", ".join(part for part in (prompt.strip().rstrip(","), suffix) if part)
               for prompt in base_prompts]
    return weights, list(zip(images, prompts)), stage2_output, abam_output, overwrite


def main():
    parser = argparse.ArgumentParser(description="Run DiffDehaze on a folder of nighttime images.")
    parser.add_argument("--config", default="configs/inference.yaml")
    parser.add_argument("--input", help="Override the input image folder.")
    parser.add_argument("--output", help="Use stage2/ and abam/ under this output folder.")
    text = parser.add_mutually_exclusive_group()
    text.add_argument("--caption-csv", help="Use the scene caption for each input from a CSV.")
    text.add_argument("--prompt", help="Use one scene caption for all inputs.")
    parser.add_argument("--device", help="Override the configured PyTorch device.")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing existing ABAM results.")
    parser.add_argument("--check", action="store_true", help="Check file paths and captions without loading models.")
    args = parser.parse_args()

    try:
        with project_path(args.config).open(encoding="utf-8") as file:
            config = yaml.safe_load(file)
        weights, jobs, stage2_output, abam_output, overwrite = plan_run(config, args)
    except (OSError, ValueError, KeyError, yaml.YAMLError) as error:
        parser.exit(2, f"{error}\n")

    print(f"Inputs: {len(jobs)}\nVAE outputs: {stage2_output}\nFinal images: {abam_output / 'images'}")
    if args.check:
        return

    import torch

    device = torch.device(args.device or config["inference"]["device"])
    if device.type == "cuda":
        if not torch.cuda.is_available():
            parser.exit(2, f"CUDA is not available for {device}.\n")
        if device.index is not None and device.index >= torch.cuda.device_count():
            parser.exit(2, f"CUDA device does not exist: {device}.\n")

    from inference_stage2 import restore_images
    from inference_abam import adjust_brightness

    restored = restore_images(config, weights, jobs, stage2_output, device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    adjust_brightness(
        config["abam"], weights["abam"], restored, abam_output, device, overwrite,
    )


if __name__ == "__main__":
    main()
