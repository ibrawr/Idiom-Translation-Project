# model_loader.py
# loads mt5-small model and tokenizer so other scripts can just import them

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "google/mt5-small"


def get_device():
    """Pick GPU if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_tokenizer(model_name=MODEL_NAME):
    """Load the mt5 tokenizer from HuggingFace."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        raise RuntimeError(f"failed to load tokenizer for '{model_name}': {e}")
    return tokenizer


def load_model(model_name=MODEL_NAME, device=None):
    """Load mt5-small and move it to the right device."""
    if device is None:
        device = get_device()
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    except Exception as e:
        raise RuntimeError(f"failed to load model '{model_name}': {e}")
    model = model.to(device)
    return model


def load_model_and_tokenizer(model_name=MODEL_NAME, device=None):
    """Load both model and tokenizer in one call. Easiest way to get started."""
    tokenizer = load_tokenizer(model_name)
    model = load_model(model_name, device)
    return model, tokenizer
