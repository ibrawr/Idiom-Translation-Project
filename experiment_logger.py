"""
experiment_logger.py

Builds the final Person 7 experiment summary by combining:
- training logs from Full FT and LoRA
- evaluation metrics from Person 6
- parameter counts from the saved checkpoints

Usage:
    python experiment_logger.py
    python experiment_logger.py --config config.yaml
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, Any, Optional

import yaml
from transformers import AutoModelForSeq2SeqLM
from peft import PeftModel


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_training_log(log_path: str) -> Dict[str, Any]:
    """
    Parse the simple text logs created by the training notebooks.
    Missing fields are returned as empty strings so the summary can still be generated.
    """
    path = Path(log_path)
    if not path.exists():
        return {
            "model_name": "",
            "method": "",
            "epochs": "",
            "batch_size": "",
            "learning_rate": "",
            "train_samples": "",
            "val_samples": "",
            "test_samples": "",
            "device": "",
            "training_runtime_seconds": "",
        }

    text = path.read_text(encoding="utf-8", errors="ignore")

    patterns = {
        "model_name": r"Model:\s*(.+)",
        "method": r"Method:\s*(.+)",
        "epochs": r"Epochs:\s*([^\n]+)",
        "batch_size": r"Batch size:\s*([^\n]+)",
        "learning_rate": r"Learning rate:\s*([^\n]+)",
        "train_samples": r"Train samples:\s*([^\n]+)",
        "val_samples": r"Validation samples:\s*([^\n]+)",
        "test_samples": r"Test samples:\s*([^\n]+)",
        "device": r"Device:\s*([^\n]+)",
        "training_runtime_seconds": r"Training runtime \(seconds\):\s*([^\n]+)",
    }

    data: Dict[str, Any] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        data[key] = match.group(1).strip() if match else ""

    return data


def _safe_int(x: Any) -> Any:
    if x == "" or x is None:
        return ""
    try:
        return int(float(x))
    except Exception:
        return x


def _safe_float(x: Any) -> Any:
    if x == "" or x is None:
        return ""
    try:
        return float(x)
    except Exception:
        return x


def count_parameters_full_ft(checkpoint_path: str) -> Dict[str, int]:
    """
    Load a fully fine-tuned model checkpoint and count:
    - total number of parameters
    - number of trainable parameters
    """
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_path)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
    }


def count_parameters_lora(base_model_name: str, adapter_path: str) -> Dict[str, int]:
    """
    Load a base model and attach the LoRA adapter.
    Then compute the total and trainable parameter counts.
    """
    base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
    model = PeftModel.from_pretrained(base_model, adapter_path)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
    }


def load_scores(scores_csv: str) -> Dict[str, Dict[str, Any]]:
    """
    Expected format:
        Metric,Full_FT,LoRA
        BLEU,0.17,0.03
        ROUGE-L,0.0,0.0
        CFS_avg,1.0,1.0
    """
    path = Path(scores_csv)
    if not path.exists():
        raise FileNotFoundError(f"Scores file not found: {path}")

    metrics = {"full_ft": {}, "lora": {}}

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            metric = row["Metric"].strip()
            metrics["full_ft"][metric] = _safe_float(row.get("Full_FT", ""))
            metrics["lora"][metric] = _safe_float(row.get("LoRA", ""))

    return metrics


def choose_default_model(scores: Dict[str, Dict[str, Any]]) -> str:
    """
    Decide which model should be considered the default serving model.
    The decision is currently based on the BLEU score.
    """
    full_bleu = scores.get("full_ft", {}).get("BLEU", 0) or 0
    lora_bleu = scores.get("lora", {}).get("BLEU", 0) or 0
    return "full_ft" if full_bleu >= lora_bleu else "lora"


def build_rows(config: Dict[str, Any]) -> list[Dict[str, Any]]:
    """
    Build the rows of the final experiment summary table by combining:
    - training log information
    - evaluation metrics
    - parameter counts
    """
    project = config["project"]
    paths = config["paths"]

    full_log = parse_training_log(paths["full_ft_training_log"])
    lora_log = parse_training_log(paths["lora_training_log"])
    scores = load_scores(paths["scores_csv"])

    # Parameter counting may take a little time because the models must be loaded.
    full_params = count_parameters_full_ft(paths["full_ft_checkpoint"])
    lora_params = count_parameters_lora(project["base_model"], paths["lora_checkpoint"])

    default_model = choose_default_model(scores) # Determine which model should be the default

    rows = [
        {
            "model_type": "full_ft",
            "base_model": project["base_model"],
            "checkpoint_path": paths["full_ft_checkpoint"],
            "epochs": _safe_int(full_log["epochs"]),
            "batch_size": _safe_int(full_log["batch_size"]),
            "learning_rate": full_log["learning_rate"],
            "train_samples": _safe_int(full_log["train_samples"]),
            "val_samples": _safe_int(full_log["val_samples"]),
            "test_samples": _safe_int(full_log["test_samples"]),
            "device": full_log["device"],
            "training_runtime_seconds": _safe_float(full_log["training_runtime_seconds"]),
            "total_parameters": full_params["total_parameters"],
            "trainable_parameters": full_params["trainable_parameters"],
            "BLEU": scores["full_ft"].get("BLEU", ""),
            "ROUGE_L": scores["full_ft"].get("ROUGE-L", ""),
            "CFS_avg": scores["full_ft"].get("CFS_avg", ""),
            "serving_role": "default" if default_model == "full_ft" else "optional",
        },
        {
            "model_type": "lora",
            "base_model": project["base_model"],
            "checkpoint_path": paths["lora_checkpoint"],
            "epochs": _safe_int(lora_log["epochs"]),
            "batch_size": _safe_int(lora_log["batch_size"]),
            "learning_rate": lora_log["learning_rate"],
            "train_samples": _safe_int(lora_log["train_samples"]),
            "val_samples": _safe_int(lora_log["val_samples"]),
            "test_samples": _safe_int(lora_log["test_samples"]),
            "device": lora_log["device"],
            "training_runtime_seconds": _safe_float(lora_log["training_runtime_seconds"]),
            "total_parameters": lora_params["total_parameters"],
            "trainable_parameters": lora_params["trainable_parameters"],
            "BLEU": scores["lora"].get("BLEU", ""),
            "ROUGE_L": scores["lora"].get("ROUGE-L", ""),
            "CFS_avg": scores["lora"].get("CFS_avg", ""),
            "serving_role": "default" if default_model == "lora" else "optional",
        },
    ]
    return rows


def save_summary(rows: list[Dict[str, Any]], output_csv: str) -> None:
    fieldnames = [
        "model_type",
        "base_model",
        "checkpoint_path",
        "epochs",
        "batch_size",
        "learning_rate",
        "train_samples",
        "val_samples",
        "test_samples",
        "device",
        "training_runtime_seconds",
        "total_parameters",
        "trainable_parameters",
        "BLEU",
        "ROUGE_L",
        "CFS_avg",
        "serving_role",
    ]

    with open(output_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    args = parser.parse_args()

    config = load_config(args.config)
    output_csv = config["paths"]["results_summary_csv"]

    rows = build_rows(config)
    save_summary(rows, output_csv)

    print(f"Saved experiment summary to: {output_csv}")
    for row in rows:
        print("-" * 60)
        for key, value in row.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
