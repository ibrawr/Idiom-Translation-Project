# training_pipeline.py
# shared training pipeline for mt5-small, used by Person 4 and Person 5

import os
import csv
import json

import torch
from datasets import Dataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)


#  validation 

def validate_dataset_schema(dataset, split_name="dataset"):
    """Make sure every record has the three required columns and they look right."""
    required = ["input_ids", "attention_mask", "labels"]

    for col in required:
        if col not in dataset.column_names:
            raise ValueError(f"{split_name} is missing column: '{col}'")

    # pull each column as a list and check every value, not just a sample
    for col in required:
        column_data = dataset[col]
        for i, val in enumerate(column_data):
            if not isinstance(val, list) or len(val) == 0:
                raise ValueError(
                    f"{split_name} record {i}: '{col}' should be a non-empty list, "
                    f"got {type(val).__name__} (len={len(val) if isinstance(val, list) else 'n/a'})"
                )

    print(f"  {split_name} schema ok - all {len(dataset)} records validated")


def build_dataset_summary(csv_path, train_count, val_count, test_count,
                          processed_files=None):
    """
    Read dataset_final.csv and build a structured summary connecting
    Person 1's source data to Person 2's processed splits.
    """
    total_processed = train_count + val_count + test_count

    summary = {
        "source_file": os.path.basename(csv_path),
        "source_path": os.path.abspath(csv_path),
        "source_available": False,
        "source_rows": 0,
        "source_columns": [],
        "category_counts": {},
        "processed_files": processed_files or {
            "train": "processed_train.json",
            "val": "processed_val.json",
            "test": "processed_test.json",
        },
        "processed_splits": {
            "train": train_count,
            "val": val_count,
            "test": test_count,
            "total": total_processed,
        },
        "counts_match": False,
        "source_samples": [],
        "warnings": [],
    }

    # resolve the source file, try without .csv as fallback
    if not os.path.isfile(csv_path):
        alt_path = os.path.splitext(csv_path)[0]
        if os.path.isfile(alt_path):
            print(f"  note: found '{os.path.basename(alt_path)}' instead of '{os.path.basename(csv_path)}'")
            csv_path = alt_path
            summary["source_file"] = os.path.basename(alt_path)
            summary["source_path"] = os.path.abspath(alt_path)
        else:
            summary["warnings"].append(
                f"{os.path.basename(csv_path)} not found (also checked "
                f"'{os.path.basename(alt_path)}')"
            )
            return summary

    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        summary["warnings"].append("csv is empty")
        return summary

    summary["source_available"] = True
    summary["source_rows"] = len(rows)
    summary["source_columns"] = list(rows[0].keys())

    expected = {"Urdu Idiom", "English Translation", "category"}
    missing = expected - set(summary["source_columns"])
    if missing:
        summary["warnings"].append(f"missing columns: {list(missing)}")

    # count idioms per category
    if "category" in rows[0]:
        cats = {}
        for row in rows:
            cat = row.get("category", "unknown").strip()
            cats[cat] = cats.get(cat, 0) + 1
        summary["category_counts"] = dict(sorted(cats.items()))

    summary["counts_match"] = (len(rows) == total_processed)
    if not summary["counts_match"]:
        diff = abs(len(rows) - total_processed)
        summary["warnings"].append(
            f"source has {len(rows)} rows but processed splits have {total_processed} "
            f"(difference of {diff}, likely filtered during tokenization by Person 2)"
        )

    # grab a few source rows so teammates can see the original data
    for row in rows[:3]:
        summary["source_samples"].append({
            "urdu_idiom": row.get("Urdu Idiom", ""),
            "english_translation": row.get("English Translation", ""),
            "category": row.get("category", ""),
        })

    return summary


def save_dataset_summary(summary, output_path):
    """Save the dataset integration summary to a json file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"saved dataset summary to {output_path}")


# --- dataset loading ---

def load_processed_dataset(filepath):
    """Read a JSONL file (one json per line) and return a HuggingFace Dataset."""
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"processed file not found: {filepath}")

    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"bad json on line {line_num} in {filepath}: {e}")

    if not records:
        raise ValueError(f"no records found in {filepath}")

    return Dataset.from_list(records)


def load_all_datasets(train_path, val_path, test_path):
    """Load all three splits and print their sizes."""
    train_dataset = load_processed_dataset(train_path)
    val_dataset = load_processed_dataset(val_path)
    test_dataset = load_processed_dataset(test_path)
    print(f"loaded {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test samples")
    return train_dataset, val_dataset, test_dataset


# --- data collator ---

def get_data_collator(tokenizer, model=None):
    """Pads batches dynamically. Labels get -100 padding so the loss ignores them."""
    return DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
    )


# training arguments 

def get_training_args(
    output_dir="./results",
    num_epochs=3,
    batch_size=4,
    learning_rate=5e-4,
    logging_steps=50,
    save_strategy="epoch",
    eval_strategy="epoch",
    generation_max_length=128,
    **kwargs,
):
    """Build Seq2SeqTrainingArguments with good defaults. Override anything via kwargs."""
    use_fp16 = torch.cuda.is_available()

    args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        eval_strategy=eval_strategy,
        save_strategy=save_strategy,
        predict_with_generate=True,
        generation_max_length=generation_max_length,
        logging_steps=logging_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        fp16=use_fp16,
        report_to="none",
        **kwargs,
    )
    return args


#  trainer builder 

def build_trainer(
    model,
    tokenizer,
    training_args,
    train_dataset,
    eval_dataset=None,
):
    """
    Create a Seq2SeqTrainer. Works with normal models and PEFT-wrapped ones,
    so Person 4 and Person 5 can both use this.
    """
    data_collator = get_data_collator(tokenizer, model)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
    )
    return trainer


# high-level train function 

def train_model(
    model,
    tokenizer,
    train_dataset,
    eval_dataset=None,
    output_dir="./results",
    num_epochs=3,
    batch_size=4,
    learning_rate=5e-4,
    resume_from_checkpoint=None,
    **kwargs,
):
    """
    One-call training: sets up args, builds trainer, runs training.
    Returns the trainer so you can save the model or run eval after.
    Pass resume_from_checkpoint=<path> to continue from a saved checkpoint.
    """
    training_args = get_training_args(
        output_dir=output_dir,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        **kwargs,
    )

    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        training_args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    return trainer
