# checkpoint_utils.py
# helpers for saving/loading checkpoints and resuming training

import os
import json


def find_latest_checkpoint(output_dir="./results"):
    """Look for the most recent checkpoint folder inside output_dir."""
    if not os.path.isdir(output_dir):
        print(f"output dir not found: {output_dir}")
        return None

    checkpoints = []
    for d in os.listdir(output_dir):
        if not os.path.isdir(os.path.join(output_dir, d)):
            continue
        if not d.startswith("checkpoint-"):
            continue
        # safely parseing the step number, also skipping folders with bad names
        try:
            step = int(d.split("-")[-1])
            checkpoints.append((d, step))
        except ValueError:
            print(f"  skipping malformed checkpoint folder: {d}")
            continue

    if not checkpoints:
        print(f"no checkpoints found in {output_dir}")
        return None

    latest = max(checkpoints, key=lambda x: x[1])
    checkpoint_path = os.path.join(output_dir, latest[0])
    print(f"latest checkpoint: {checkpoint_path}")
    return checkpoint_path


def save_checkpoint_info(output_dir, info_dict):
    """Save a small json with info about the training run (hyperparams, notes, etc.)."""
    os.makedirs(output_dir, exist_ok=True)
    info_path = os.path.join(output_dir, "checkpoint_info.json")
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info_dict, f, indent=2, ensure_ascii=False)
    print(f"saved checkpoint info to {info_path}")


def load_checkpoint_info(output_dir):
    """Load the checkpoint_info.json we saved earlier, or None if it doesn't exist."""
    info_path = os.path.join(output_dir, "checkpoint_info.json")
    if not os.path.exists(info_path):
        print(f"no checkpoint info at {info_path}")
        return None
    with open(info_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_resume_checkpoint(output_dir="./results"):
    """
    Check if we can resume from a previous run.
    Returns the checkpoint path if one exists, None otherwise.
    """
    ckpt = find_latest_checkpoint(output_dir)
    if ckpt:
        print(f"can resume training from: {ckpt}")
    else:
        print("no previous checkpoint, will train from scratch")
    return ckpt
