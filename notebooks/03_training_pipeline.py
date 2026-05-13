# 03_training_pipeline.py
# Person 3 - main script, runs the whole pipeline end to end
# run from project root:  python notebooks/03_training_pipeline.py

import sys
import os

# add project root to path so we can import from src/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.model_loader import load_model_and_tokenizer, get_device
from src.training_pipeline import (
    load_all_datasets,
    train_model,
    validate_dataset_schema,
    build_dataset_summary,
    save_dataset_summary,
)
from src.checkpoint_utils import save_checkpoint_info, get_resume_checkpoint
from src.inference_utils import batch_inference, save_predictions


# config
# note for team, change these if your folder layout is different

DATASET_CSV_PATH = os.path.join(PROJECT_ROOT, "dataset_final.csv")
TRAIN_PATH       = os.path.join(PROJECT_ROOT, "processed_train.json")
VAL_PATH         = os.path.join(PROJECT_ROOT, "processed_val.json")
TEST_PATH        = os.path.join(PROJECT_ROOT, "processed_test.json")

OUTPUT_DIR       = os.path.join(PROJECT_ROOT, "results")
PREDICTIONS_PATH = os.path.join(PROJECT_ROOT, "predictions.json")
SUMMARY_PATH     = os.path.join(OUTPUT_DIR, "dataset_integration_summary.json")

SANITY_EPOCHS = 1
SANITY_BATCH  = 2
LEARNING_RATE = 5e-4
LOGGING_STEPS = 20

# keep this False for a normal fresh run.
# only set to True if you want to continue from a previous training checkpoint.
RESUME_FROM_CHECKPOINT = False


# load model

print("loading model and tokenizer...")

device = get_device()
print(f"using device: {device}")

model, tokenizer = load_model_and_tokenizer()
print(f"model: google/mt5-small")
print(f"vocab size: {tokenizer.vocab_size}")
print(f"total params: {sum(p.numel() for p in model.parameters()):,}")


#  load processed datasets (Person 2 output)

print("\nloading processed json files...")

train_dataset, val_dataset, test_dataset = load_all_datasets(
    TRAIN_PATH, VAL_PATH, TEST_PATH
)

print(f"columns: {train_dataset.column_names}")


# validate schema of processed files

print("\nvalidating processed data schema...")

validate_dataset_schema(train_dataset, "train")
validate_dataset_schema(val_dataset, "val")
validate_dataset_schema(test_dataset, "test")


# integrate Person 1 output (dataset_final.csv)
# this loads the original source dataset, reads its content (categories,
# sample rows, row count), and compares it with the processed splits so
# we have a clear record that Person 1 -> Person 2 -> Person 3 are linked.

print("\nbuilding dataset integration summary...")

summary = build_dataset_summary(
    DATASET_CSV_PATH,
    train_count=len(train_dataset),
    val_count=len(val_dataset),
    test_count=len(test_dataset),
    processed_files={
        "train": os.path.basename(TRAIN_PATH),
        "val": os.path.basename(VAL_PATH),
        "test": os.path.basename(TEST_PATH),
    },
)

if summary["source_available"]:
    print(f"  source: {summary['source_file']} ({summary['source_rows']} rows)")
    print(f"  columns: {summary['source_columns']}")
    print(f"  categories: {summary['category_counts']}")
    print(f"  processed total: {summary['processed_splits']['total']}")
    if summary["counts_match"]:
        print("  row counts match between source and processed data")
    for w in summary["warnings"]:
        print(f"  warning: {w}")

    # show a couple of original idioms so we can see the raw data
    print("\n  sample source rows from dataset_final.csv:")
    for s in summary["source_samples"]:
        print(f"    english: {s['english_translation']}")
        print(f"    urdu:    {s['urdu_idiom']}")
        print(f"    category: {s['category']}")
        print()
else:
    print("  source csv not available, skipping")
    for w in summary["warnings"]:
        print(f"  warning: {w}")

# save the summary so Person 4/5/6/7 can see the data lineage
save_dataset_summary(summary, SUMMARY_PATH)


#inspectin one sample 

if len(train_dataset) == 0:
    raise ValueError("train split is empty, cannot preview sample or train model")

print("\nchecking first training sample...")

sample = train_dataset[0]
print(f"input_ids length:      {len(sample['input_ids'])}")
print(f"attention_mask length: {len(sample['attention_mask'])}")
print(f"labels length:         {len(sample['labels'])}")

decoded_input = tokenizer.decode(sample["input_ids"], skip_special_tokens=True)
clean_labels = [t for t in sample["labels"] if t != -100]
decoded_label = tokenizer.decode(clean_labels, skip_special_tokens=True)
print(f"\ninput:  {decoded_input}")
print(f"target: {decoded_label}")


#  checkpoint resume check 

resume_ckpt = None
if RESUME_FROM_CHECKPOINT:
    resume_ckpt = get_resume_checkpoint(OUTPUT_DIR)
    if resume_ckpt:
        print(f"\nresuming from checkpoint: {resume_ckpt}")
    else:
        print("\nno checkpoint found, starting fresh")
else:
    print("\nstarting fresh training run")


#  sanity training run 
# this shared pipeline is designed so Person 4 (full fine-tuning) and
# Person 5 (LoRA/PEFT) can both reuse it with their own settings.
# here we just do a quick sanity run to make sure everything works.

print(f"training config: epochs={SANITY_EPOCHS}, batch={SANITY_BATCH}")

# only pass resume_from_checkpoint when we actually have a path,
# otherwise let the trainer start completely fresh
train_kwargs = dict(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    output_dir=OUTPUT_DIR,
    num_epochs=SANITY_EPOCHS,
    batch_size=SANITY_BATCH,
    learning_rate=LEARNING_RATE,
    logging_steps=LOGGING_STEPS,
)
if resume_ckpt:
    train_kwargs["resume_from_checkpoint"] = resume_ckpt

trainer = train_model(**train_kwargs)

print("training done.")


# save checkpoint info 

print("\nsaving checkpoint info...")

save_checkpoint_info(OUTPUT_DIR, {
    "model_name": "google/mt5-small",
    "task": "English Idiom to Urdu Cultural Translation",
    "person": "Person 3 - Core Training Pipeline",
    "epochs_trained": SANITY_EPOCHS,
    "batch_size": SANITY_BATCH,
    "learning_rate": LEARNING_RATE,
    "train_samples": len(train_dataset),
    "val_samples": len(val_dataset),
    "test_samples": len(test_dataset),
    "dependencies": {
        "person_1": {
            "file": summary["source_file"],
            "rows": summary["source_rows"],
            "script": "01_dataset_preparation.py",
        },
        "person_2": {
            "files": summary["processed_files"],
            "total_samples": summary["processed_splits"]["total"],
        },
    },
    "dataset_summary_file": "dataset_integration_summary.json",
    "note": "sanity run to verify the pipeline works end to end",
})


# inference on test samples 

print("\nrunning inference on test samples...")

num_samples = min(5, len(test_dataset))
test_subset = test_dataset.select(range(num_samples))

predictions, references = batch_inference(
    model, tokenizer, test_subset, batch_size=num_samples,
)

# decode the input prompts for display and saving
input_texts = [
    tokenizer.decode(test_subset[i]["input_ids"], skip_special_tokens=True)
    for i in range(num_samples)
]

print("\nresults:")
for i in range(num_samples):
    print(f"\n  [{i + 1}]")
    print(f"  input:      {input_texts[i]}")
    print(f"  reference:  {references[i]}")
    print(f"  prediction: {predictions[i]}")


# save predictions-

run_info = {
    "model": "google/mt5-small",
    "task": "English Idiom to Urdu Cultural Translation",
    "person": "Person 3",
    "source_dataset": summary["source_file"],
    "test_split": os.path.basename(TEST_PATH),
    "num_samples": num_samples,
}

save_predictions(predictions, references, PREDICTIONS_PATH,
                 input_texts=input_texts, run_info=run_info)

print("\nrun finished. predictions saved.")

# - how teammates should reuse this pipeline
#
# Person 4 (full fine-tuning):
#   from src.model_loader import load_model_and_tokenizer
#   from src.training_pipeline import load_all_datasets, train_model
#   model, tokenizer = load_model_and_tokenizer()
#   train_ds, val_ds, test_ds = load_all_datasets(...)
#   trainer = train_model(model, tokenizer, train_ds, val_ds,
#                         output_dir="./results_full", num_epochs=10)
#
# Person 5 (LoRA / PEFT):
#   from peft import LoraConfig, get_peft_model, TaskType
#   from src.model_loader import load_model_and_tokenizer
#   from src.training_pipeline import load_all_datasets, get_training_args, build_trainer
#   model, tokenizer = load_model_and_tokenizer()
#   model = get_peft_model(model, LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, r=8, ...))
#   args = get_training_args(output_dir="./results_lora", num_epochs=10)
#   trainer = build_trainer(model, tokenizer, args, train_ds, val_ds)
#   trainer.train()
