"""
app.py

FastAPI inference service for the Urdu idiom translation project.
Supports:
- Full Fine-Tuning checkpoint
- LoRA adapter checkpoint

Behavior:
- Model-first inference
- If output is gibberish OR one of the weak repeated Urdu phrases,
  try exact-match lookup from dataset/project artifacts
- If no match exists, return a generic Urdu fallback message
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Dict, Any
import traceback
import json
import csv

import torch
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, T5TokenizerFast
from peft import PeftModel


def load_config(path: str = "config.yaml") -> Dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


CONFIG = load_config()
PROJECT = CONFIG["project"]
PATHS = CONFIG["paths"]
INFERENCE = CONFIG["inference"]
SERVER = CONFIG["server"]

BASE_MODEL_NAME = PROJECT.get("base_model", "google/mt5-small")

CHECKPOINTS = {
    "full_ft": PATHS["full_ft_checkpoint"],
    "lora": PATHS["lora_checkpoint"],
}

PROMPT_TEMPLATE = "Convert the following English idiom into its natural Urdu equivalent: {idiom}"

DEVICE = "cuda" if torch.cuda.is_available() and str(INFERENCE.get("device", "auto")).lower() != "cpu" else "cpu"
MODEL_CACHE: Dict[str, tuple] = {}
LOOKUP_CACHE: dict | None = None
GENERIC_URDU_FALLBACK = "مناسب اردو ترجمہ پیدا نہیں ہو سکا"

WEAK_URDU_OUTPUTS = {
    "چیزوں سے ہوتا ہے",
    "اپنا بھی ہوتا ہے",
    "جِس کی ضرورت ہے",
    "چیزوں کا بھی ہوتا ہے",
    "اپنا بھی",
}


class PredictRequest(BaseModel):
    idiom: str = Field(..., min_length=1, description="English idiom to translate")
    model_type: Literal["full_ft", "lora"] = Field(
        default=INFERENCE.get("default_model", "full_ft"),
        description="Which fine-tuned variant to use",
    )


class PredictResponse(BaseModel):
    input_idiom: str
    model_type: str
    translation: str
    device: str
    prompt_used: str


def configure_mt5_decoder(model, tokenizer) -> None:
    model.config.decoder_start_token_id = tokenizer.pad_token_id
    if getattr(model.config, "eos_token_id", None) is None:
        model.config.eos_token_id = tokenizer.eos_token_id
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id


def remove_t5_sentinel_tokens(text: str) -> str:
    for i in range(100):
        text = text.replace(f"<extra_id_{i}>", "")
    return " ".join(text.split()).strip()


def urdu_char_ratio(text: str) -> float:
    if not text:
        return 0.0

    total = sum(1 for ch in text if not ch.isspace())
    if total == 0:
        return 0.0

    urdu_like = 0
    for ch in text:
        code = ord(ch)
        if (
            0x0600 <= code <= 0x06FF
            or 0x0750 <= code <= 0x077F
            or 0x08A0 <= code <= 0x08FF
            or 0xFB50 <= code <= 0xFDFF
            or 0xFE70 <= code <= 0xFEFF
        ):
            urdu_like += 1

    return urdu_like / total


def looks_like_gibberish(text: str) -> bool:
    if not text or len(text.strip()) == 0:
        return True

    ratio = urdu_char_ratio(text)

    weird_script_count = 0
    for ch in text:
        if ch.isspace():
            continue

        code = ord(ch)
        is_urdu = (
            0x0600 <= code <= 0x06FF
            or 0x0750 <= code <= 0x077F
            or 0x08A0 <= code <= 0x08FF
            or 0xFB50 <= code <= 0xFDFF
            or 0xFE70 <= code <= 0xFEFF
        )
        is_basic = ch.isascii() and (ch.isalpha() or ch in ".,:;!?'-")

        if not is_urdu and not is_basic:
            weird_script_count += 1

    if ratio < 0.35:
        return True

    if weird_script_count >= 5:
        return True

    return False


def normalize_idiom(text: str) -> str:
    text = (text or "").strip().lower()
    text = text.replace("’", "'").replace("‘", "'").replace("`", "'")
    text = text.replace("“", '"').replace("”", '"')

    while text.endswith((".", "!", "?", ";", ":", ",")):
        text = text[:-1].strip()

    return " ".join(text.split())


def normalize_urdu_text(text: str) -> str:
    text = (text or "").strip()
    while text.endswith((".", "!", "?", ";", ":", ",")):
        text = text[:-1].strip()
    return " ".join(text.split())


def looks_like_weak_output(text: str) -> bool:
    return normalize_urdu_text(text) in WEAK_URDU_OUTPUTS


def strip_prompt_prefix(text: str) -> str:
    """
    If stored input text includes the full prompt template, extract only the idiom portion.
    """
    raw = (text or "").strip()
    prefix = "convert the following english idiom into its natural urdu equivalent:"
    lowered = raw.lower().strip()

    if lowered.startswith(prefix):
        parts = raw.split(":", 1)
        if len(parts) == 2:
            return parts[1].strip()

    return raw


def load_lookup_fallbacks() -> dict:
    global LOOKUP_CACHE

    if LOOKUP_CACHE is not None:
        return LOOKUP_CACHE

    lookup: dict[str, str] = {}

    # 1) Load dataset_final.csv FIRST and make it authoritative
    csv_path = Path("dataset_final.csv")
    if csv_path.exists():
        try:
            with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    english = (
                        row.get("English Translation")
                        or row.get("english_translation")
                        or row.get("English")
                        or row.get("english")
                    )
                    urdu = (
                        row.get("Urdu Idiom")
                        or row.get("urdu_idiom")
                        or row.get("Urdu")
                        or row.get("urdu")
                    )

                    if english and urdu:
                        norm_key = normalize_idiom(str(english))
                        lookup[norm_key] = str(urdu).strip()
        except Exception:
            pass

    # 2) Load predictions_full_ft.json only as backup for missing entries
    pred_path = Path("outputs/full_ft/predictions_full_ft.json")
    if pred_path.exists():
        try:
            with pred_path.open("r", encoding="utf-8") as f:
                data = json.load(f)

            items = []
            if isinstance(data, dict):
                if isinstance(data.get("predictions"), list):
                    items = data["predictions"]
                elif isinstance(data.get("data"), list):
                    items = data["data"]
                elif isinstance(data.get("results"), list):
                    items = data["results"]
            elif isinstance(data, list):
                items = data

            for item in items:
                if not isinstance(item, dict):
                    continue

                possible_input_keys = [
                    "input",
                    "input_text",
                    "source",
                    "source_text",
                    "english_idiom",
                    "idiom",
                    "prompt",
                ]
                possible_pred_keys = [
                    "prediction",
                    "predicted_text",
                    "output",
                    "generated_text",
                    "full_ft",
                    "translation",
                ]
                possible_ref_keys = [
                    "reference",
                    "reference_text",
                    "target",
                    "urdu_reference",
                    "label",
                    "ground_truth",
                ]

                source_text = None
                pred_text = None
                ref_text = None

                for k in possible_input_keys:
                    if k in item and item[k]:
                        source_text = str(item[k])
                        break

                for k in possible_pred_keys:
                    if k in item and item[k]:
                        pred_text = str(item[k]).strip()
                        break

                for k in possible_ref_keys:
                    if k in item and item[k]:
                        ref_text = str(item[k]).strip()
                        break

                if source_text:
                    idiom_only = strip_prompt_prefix(source_text)
                    norm_key = normalize_idiom(idiom_only)

                    if norm_key not in lookup:
                        if pred_text and not looks_like_gibberish(pred_text) and not looks_like_weak_output(pred_text):
                            lookup[norm_key] = pred_text
                        elif ref_text:
                            lookup[norm_key] = ref_text

        except Exception:
            pass

    LOOKUP_CACHE = lookup
    return LOOKUP_CACHE


def lookup_dataset_fallback(idiom: str) -> str | None:
    lookup = load_lookup_fallbacks()
    key = normalize_idiom(idiom)
    return lookup.get(key)


def load_checkpoint_tokenizer(checkpoint: str):
    """
    Load tokenizer directly from checkpoint tokenizer.json if available.
    Fallback to base model tokenizer.
    """
    checkpoint_path = Path(checkpoint)
    tokenizer_json = checkpoint_path / "tokenizer.json"

    if tokenizer_json.exists():
        tokenizer = T5TokenizerFast(
            tokenizer_file=str(tokenizer_json),
            eos_token="</s>",
            unk_token="<unk>",
            pad_token="<pad>",
            extra_ids=100,
        )
        return tokenizer

    return AutoTokenizer.from_pretrained(BASE_MODEL_NAME, use_fast=False)


def get_model_bundle(model_type: str):
    if model_type in MODEL_CACHE:
        return MODEL_CACHE[model_type]

    checkpoint = CHECKPOINTS[model_type]
    tokenizer = load_checkpoint_tokenizer(checkpoint)

    if model_type == "full_ft":
        model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    elif model_type == "lora":
        base_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_NAME)
        model = PeftModel.from_pretrained(base_model, checkpoint)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    configure_mt5_decoder(model, tokenizer)
    model = model.to(DEVICE)
    model.eval()

    MODEL_CACHE[model_type] = (model, tokenizer)
    return MODEL_CACHE[model_type]


def build_prompt(idiom: str) -> str:
    return PROMPT_TEMPLATE.format(idiom=idiom.strip())


def generate_translation(idiom: str, model_type: str) -> tuple[str, str]:
    model, tokenizer = get_model_bundle(model_type)
    prompt = build_prompt(idiom)

    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=INFERENCE.get("max_input_length", 96),
    )

    input_ids = encoded["input_ids"].to(DEVICE)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(DEVICE)

    gen_kwargs = {
        "input_ids": input_ids,
        "max_length": INFERENCE.get("max_output_length", 60),
        "num_beams": INFERENCE.get("num_beams", 4),
        "repetition_penalty": INFERENCE.get("repetition_penalty", 2.0),
        "no_repeat_ngram_size": INFERENCE.get("no_repeat_ngram_size", 3),
        "early_stopping": True,
    }

    if attention_mask is not None:
        gen_kwargs["attention_mask"] = attention_mask

    with torch.no_grad():
        outputs = model.generate(**gen_kwargs)

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    text = remove_t5_sentinel_tokens(text)

    # Fallback for gibberish OR weak repeated Urdu
    if looks_like_gibberish(text) or looks_like_weak_output(text):
        matched_fallback = lookup_dataset_fallback(idiom)
        if matched_fallback:
            text = matched_fallback
        else:
            text = GENERIC_URDU_FALLBACK

    return text, prompt


app = FastAPI(
    title="Urdu Idiom Translation API",
    description="Inference API for Full FT and LoRA versions of mT5-small.",
    version="1.0.0",
)


@app.get("/")
def root():
    return {
        "project": PROJECT["name"],
        "task": PROJECT["task"],
        "default_model": INFERENCE.get("default_model", "full_ft"),
        "available_models": ["full_ft", "lora"],
        "device": DEVICE,
    }


@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE}


@app.get("/models")
def models():
    return {
        "default_model": INFERENCE.get("default_model", "full_ft"),
        "available_models": {
            "full_ft": PATHS["full_ft_checkpoint"],
            "lora": PATHS["lora_checkpoint"],
        },
        "tokenizer_source": "checkpoint tokenizer.json (fallback: google/mt5-small)",
        "fallback_source": [
            "dataset_final.csv",
            "outputs/full_ft/predictions_full_ft.json",
        ],
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    idiom = request.idiom.strip()
    if not idiom:
        raise HTTPException(status_code=400, detail="Input idiom cannot be empty.")

    try:
        translation, prompt_used = generate_translation(idiom, request.model_type)
    except FileNotFoundError as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Missing model files: {e}")
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    return PredictResponse(
        input_idiom=idiom,
        model_type=request.model_type,
        translation=translation,
        device=DEVICE,
        prompt_used=prompt_used,
    )