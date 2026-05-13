# Person 5 - LoRA / PEFT Training + Inference

import sys
import os
import time
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)

from peft import LoraConfig, get_peft_model, TaskType
from src.model_loader import load_model_and_tokenizer, get_device
from src.training_pipeline import load_all_datasets, train_model
from src.inference_utils import batch_inference, save_predictions


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_NAME = "google/mt5-small"

NUM_EPOCHS = 6
BATCH_SIZE = 8
LEARNING_RATE = 1e-4


# Dataset paths
TRAIN_PATH = os.path.join(PROJECT_ROOT, "processed_train.json")
VAL_PATH   = os.path.join(PROJECT_ROOT, "processed_val.json")
TEST_PATH  = os.path.join(PROJECT_ROOT, "processed_test.json")


# Output paths
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "lora")
PRED_PATH  = os.path.join(OUTPUT_DIR, "predictions_lora.json")
LOG_PATH   = os.path.join(OUTPUT_DIR, "training_logs.txt")

os.makedirs(OUTPUT_DIR, exist_ok=True)


print("\n==============================")
print("MT5 LoRA Training Pipeline")
print("==============================\n")


# --------------------------------------------------
# Device
# --------------------------------------------------

print("Loading device...")
device = get_device()
print("Device:", device)


# --------------------------------------------------
# Load Base Model
# --------------------------------------------------

print("\nLoading base model:", MODEL_NAME)

model, tokenizer = load_model_and_tokenizer()

# important for MT5 decoding stability
model.config.decoder_start_token_id = tokenizer.pad_token_id


# --------------------------------------------------
# Apply LoRA
# --------------------------------------------------

print("\nApplying LoRA adapters...")

lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none"
)

model = get_peft_model(model, lora_config)

print("\nTrainable parameter summary:")
model.print_trainable_parameters()


# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

print("\nLoading processed datasets...")

train_dataset, val_dataset, test_dataset = load_all_datasets(
    TRAIN_PATH,
    VAL_PATH,
    TEST_PATH
)

print("\nDataset sizes:")
print("Train:", len(train_dataset))
print("Validation:", len(val_dataset))
print("Test:", len(test_dataset))


# --------------------------------------------------
# Training
# --------------------------------------------------

print("\nStarting LoRA training...")

train_start_time = time.time()

trainer = train_model(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    output_dir=OUTPUT_DIR,
    num_epochs=NUM_EPOCHS,
    batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE
)

train_runtime = time.time() - train_start_time

print("\nTraining finished.")
print("Training time:", round(train_runtime, 2), "seconds")


# --------------------------------------------------
# Save Training Log
# --------------------------------------------------

print("\nSaving training log...")

with open(LOG_PATH, "w", encoding="utf-8") as f:

    f.write("=== LoRA Training Log ===\n\n")

    f.write(f"Date: {datetime.now()}\n\n")

    f.write("Model Information\n")
    f.write("-----------------\n")
    f.write(f"Model: {MODEL_NAME}\n")
    f.write("Method: LoRA / PEFT\n")
    f.write("Task: English Idiom → Urdu Cultural Translation\n\n")

    f.write("Training Parameters\n")
    f.write("-------------------\n")
    f.write(f"Epochs: {NUM_EPOCHS}\n")
    f.write(f"Batch size: {BATCH_SIZE}\n")
    f.write(f"Learning rate: {LEARNING_RATE}\n\n")

    f.write("Dataset Sizes\n")
    f.write("-------------\n")
    f.write(f"Train samples: {len(train_dataset)}\n")
    f.write(f"Validation samples: {len(val_dataset)}\n")
    f.write(f"Test samples: {len(test_dataset)}\n\n")

    f.write("System Information\n")
    f.write("------------------\n")
    f.write(f"Device: {device}\n")
    f.write(f"Training runtime (seconds): {round(train_runtime,2)}\n\n")

    f.write("Notes\n")
    f.write("-----\n")
    f.write("Model trained using LoRA adapters on MT5-small.\n")
    f.write("Evaluation performed on the full test set.\n")

print("Training log saved to:", LOG_PATH)


# --------------------------------------------------
# Inference
# --------------------------------------------------

print("\nRunning inference on full test set...")

predictions, references = batch_inference(
    model,
    tokenizer,
    test_dataset,
    batch_size=8
)


input_texts = [
    tokenizer.decode(test_dataset[i]["input_ids"], skip_special_tokens=True)
    for i in range(len(test_dataset))
]


# --------------------------------------------------
# Save Predictions
# --------------------------------------------------

save_predictions(
    predictions,
    references,
    PRED_PATH,
    input_texts=input_texts,
    run_info={
        "model": MODEL_NAME,
        "method": "LoRA",
        "task": "English Idiom → Urdu Cultural Translation",
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "test_samples": len(test_dataset)
    }
)

print("\nPredictions saved to:", PRED_PATH)
print("\nMT5 LoRA training and inference pipeline completed.\n")