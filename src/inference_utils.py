# inference_utils.py
# functions for generating predictions and saving them to json

import json
import torch


def get_blocked_tokens(tokenizer):
    """
    Block common T5 sentinel tokens like <extra_id_0>, <extra_id_1>, etc.
    These sometimes appear during generation if not explicitly prevented.
    """
    blocked = []

    for i in range(10):
        token = f"<extra_id_{i}>"
        token_id = tokenizer.convert_tokens_to_ids(token)

        if token_id != tokenizer.unk_token_id and token_id is not None:
            blocked.append([token_id])

    return blocked


def generate_predictions(
    model,
    tokenizer,
    input_ids_list,
    max_length=60,
    num_beams=4,
    device=None,
):
    """Generate decoded text predictions for a list of tokenized inputs."""

    if device is None:
        device = next(model.parameters()).device

    model.eval()
    predictions = []

    bad_words_ids = get_blocked_tokens(tokenizer)

    with torch.no_grad():
        for input_ids in input_ids_list:

            inputs = torch.tensor([input_ids], dtype=torch.long).to(device)

            gen_kwargs = dict(
                input_ids=inputs,
                max_length=max_length,
                num_beams=num_beams,
                repetition_penalty=2.0,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )

            if bad_words_ids:
                gen_kwargs["bad_words_ids"] = bad_words_ids

            outputs = model.generate(**gen_kwargs)

            decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
            predictions.append(decoded)

    return predictions


def batch_inference(
    model,
    tokenizer,
    dataset,
    max_length=60,
    num_beams=4,
    batch_size=8,
    device=None,
):
    """Run inference on a HuggingFace Dataset. Returns (predictions, references)."""

    if device is None:
        device = next(model.parameters()).device

    model.eval()

    predictions = []
    references = []

    bad_words_ids = get_blocked_tokens(tokenizer)

    input_ids_all = dataset["input_ids"]

    # decode reference labels
    if "labels" in dataset.column_names:
        for labels in dataset["labels"]:
            clean = [t for t in labels if t != -100]

            references.append(
                tokenizer.decode(clean, skip_special_tokens=True)
            )

    # run inference in batches
    for i in range(0, len(input_ids_all), batch_size):

        batch_ids = input_ids_all[i: i + batch_size]

        max_len = max(len(ids) for ids in batch_ids)

        padded = [
            ids + [tokenizer.pad_token_id] * (max_len - len(ids))
            for ids in batch_ids
        ]

        attention = [
            [1] * len(ids) + [0] * (max_len - len(ids))
            for ids in batch_ids
        ]

        input_tensor = torch.tensor(padded, dtype=torch.long).to(device)
        attn_tensor = torch.tensor(attention, dtype=torch.long).to(device)

        gen_kwargs = dict(
            input_ids=input_tensor,
            attention_mask=attn_tensor,
            max_length=max_length,
            num_beams=num_beams,
            repetition_penalty=2.0,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )

        if bad_words_ids:
            gen_kwargs["bad_words_ids"] = bad_words_ids

        with torch.no_grad():
            outputs = model.generate(**gen_kwargs)

        for output in outputs:
            decoded = tokenizer.decode(output, skip_special_tokens=True)
            decoded = decoded.replace("<extra_id_0>", "").strip()
            predictions.append(decoded)

    return predictions, references


def save_predictions(predictions, references, output_path,
                     input_texts=None, run_info=None):
    """
    Save predictions to JSON.
    Each entry contains: index, input, reference, prediction.
    """

    entries = []

    for i, pred in enumerate(predictions):

        entry = {"index": i}

        if input_texts and i < len(input_texts):
            entry["input"] = input_texts[i]

        if i < len(references):
            entry["reference"] = references[i]

        entry["prediction"] = pred
        entries.append(entry)

    output = {}

    if run_info:
        output["run_info"] = run_info

    output["predictions"] = entries

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"saved {len(entries)} predictions to {output_path}")