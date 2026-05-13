import os
import sys
import time
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.model_loader import load_model_and_tokenizer, get_device
from src.training_pipeline import load_all_datasets, train_model
from src.inference_utils import batch_inference, save_predictions


# -------------------------
# Configuration
# -------------------------

MODEL_NAME = "google/mt5-small"

# Slightly stronger training than LoRA
NUM_EPOCHS = 5
BATCH_SIZE = 6
LEARNING_RATE = 5e-5


TRAIN_PATH = os.path.join(PROJECT_ROOT, "processed_train.json")
VAL_PATH   = os.path.join(PROJECT_ROOT, "processed_val.json")
TEST_PATH  = os.path.join(PROJECT_ROOT, "processed_test.json")


OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "full_ft")
PRED_PATH  = os.path.join(OUTPUT_DIR, "predictions_full_ft.json")
LOG_PATH   = os.path.join(OUTPUT_DIR, "training_logs.txt")

os.makedirs(OUTPUT_DIR, exist_ok=True)


print("\n==============================")
print("Full Fine-Tuning Training")
print("==============================\n")


# -------------------------
# Device
# -------------------------

device = get_device()
print("Using device:", device)


# -------------------------
# Load model
# -------------------------

print("\nLoading base model:", MODEL_NAME)

model, tokenizer = load_model_and_tokenizer()

# Important for MT5 generation stability
model.config.decoder_start_token_id = tokenizer.pad_token_id


# -------------------------
# Load dataset
# -------------------------

print("\nLoading datasets...")

train_dataset, val_dataset, test_dataset = load_all_datasets(
    TRAIN_PATH,
    VAL_PATH,
    TEST_PATH
)

print("\nDataset sizes:")
print("Train:", len(train_dataset))
print("Validation:", len(val_dataset))
print("Test:", len(test_dataset))


# -------------------------
# Training
# -------------------------

print("\nStarting Full Fine-Tuning...\n")

train_start = time.time()

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

train_runtime = time.time() - train_start

print("\nTraining finished.")
print("Training time:", round(train_runtime, 2), "seconds")


# -------------------------
# Save log
# -------------------------

print("\nSaving training log...")

with open(LOG_PATH, "w", encoding="utf-8") as f:

    f.write("=== Full Fine-Tuning Training Log ===\n\n")

    f.write(f"Date: {datetime.now()}\n\n")

    f.write("Model Information\n")
    f.write("-----------------\n")
    f.write(f"Model: {MODEL_NAME}\n")
    f.write("Method: Full Fine-Tuning\n")
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
    f.write(f"Training runtime (seconds): {round(train_runtime,2)}\n")


print("Training log saved to:", LOG_PATH)


# -------------------------
# Inference
# -------------------------

print("\nRunning inference on test set...")

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


save_predictions(
    predictions,
    references,
    PRED_PATH,
    input_texts=input_texts,
    run_info={
        "model": MODEL_NAME,
        "method": "Full Fine-Tuning",
        "task": "English Idiom → Urdu Cultural Translation",
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "test_samples": len(test_dataset)
    }
)

print("\nPredictions saved to:", PRED_PATH)
print("\nFull Fine-Tuning training and evaluation completed.\n")