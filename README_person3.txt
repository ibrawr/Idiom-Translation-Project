Person 3 - Handoff Guide
========================

hey team, here's what i built and how to use it.


how the data flows through the project
---------------------------------------

Person 1 (01_dataset_preparation.py)
  takes the raw Excel file and produces dataset_final.csv.
  adds categories, cleans up the data, creates the base dataset.
  output: dataset_final.csv

Person 2 (prompt engineering + tokenization)
  takes dataset_final.csv and tokenizes it with mt5-small.
  creates prompt-formatted inputs, splits into train/val/test.
  output: processed_train.json, processed_val.json, processed_test.json

Person 3 (my code - core training pipeline)
  loads Person 2's processed files as HuggingFace datasets.
  validates schema of every record in all three splits.
  loads Person 1's dataset_final.csv to verify the source data
  and builds a dataset integration summary linking all three stages.
  checks for empty splits before training.
  trains mt5-small using Seq2SeqTrainer.
  runs batch inference and saves predictions with run metadata.
  saves checkpoint metadata with a full dependency chain.

Person 4 and Person 5 reuse my pipeline code to do their own training.


what each file does
-------------------

src/model_loader.py
  loads google/mt5-small model and tokenizer, picks GPU or CPU.
  main function: load_model_and_tokenizer()

src/training_pipeline.py
  core pipeline module with these functions:
  - validate_dataset_schema() - checks every record in a dataset
  - build_dataset_summary() - reads Person 1's CSV and compares with
    Person 2's processed splits, returns a summary dict
  - save_dataset_summary() - writes the summary to json
  - load_all_datasets() - loads the three processed json files
  - get_training_args() - builds Seq2SeqTrainingArguments
  - build_trainer() - creates a Seq2SeqTrainer (works with PEFT too)
  - train_model() - does everything in one call (args + trainer + train)

src/checkpoint_utils.py
  - find_latest_checkpoint() - finds most recent checkpoint folder,
    safely skips malformed folder names
  - save_checkpoint_info() - saves run metadata to json
  - load_checkpoint_info() - reads it back
  - get_resume_checkpoint() - shortcut that finds and returns a
    checkpoint path or None

src/inference_utils.py
  - generate_predictions() - single-sample inference from token ids
  - batch_inference() - runs inference on a HuggingFace Dataset,
    returns (predictions, references)
  - save_predictions() - writes results to json, optionally includes
    a run_info header for traceability

notebooks/03_training_pipeline.py
  main demo script that runs the whole pipeline:
  1. loads mt5-small model and tokenizer
  2. loads Person 2's processed train/val/test files
  3. validates schema of every record in all splits
  4. loads Person 1's dataset_final.csv and builds an integration
     summary (categories, row counts, source samples)
  5. saves the summary to results/dataset_integration_summary.json
  6. checks that train split is not empty
  7. previews one sample (filters -100 from labels when decoding)
  8. checks RESUME_FROM_CHECKPOINT flag before training
  9. runs a 1-epoch sanity training run
  10. saves checkpoint_info.json with dependency chain
  11. runs batch inference on 5 test samples
  12. saves predictions.json with run_info metadata

  config flags at the top of the file:
  - RESUME_FROM_CHECKPOINT = False (set to True to resume old run)
  - SANITY_EPOCHS, SANITY_BATCH, LEARNING_RATE, LOGGING_STEPS


files that must stay in the project root
-----------------------------------------

these are the team files my code depends on:

  dataset_final.csv          Person 1 output (source dataset)
  01_dataset_preparation.py  Person 1 script
  processed_train.json       Person 2 output (tokenized train split)
  processed_val.json         Person 2 output (tokenized val split)
  processed_test.json        Person 2 output (tokenized test split)

the src/ folder with __init__.py must stay as is.

after running the pipeline, the results/ folder will contain:

  dataset_integration_summary.json
    links Person 1 source data and Person 2 processed splits.
    includes resolved source path, category counts, source samples,
    processed file names, and row count comparison.

  checkpoint_info.json
    training run metadata with a dependencies section showing
    which Person 1 and Person 2 files were used.

predictions.json (in project root)
  model predictions with a run_info header showing model name,
  source dataset, and test split used. each prediction entry has
  index, decoded input, reference, and model prediction.


for Person 4 - full fine-tuning
-------------------------------

import my modules and call train_model() directly.
use the same path setup as 03_training_pipeline.py so file paths
always resolve correctly no matter where you run from:

  import sys, os

  PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
  sys.path.insert(0, PROJECT_ROOT)

  from src.model_loader import load_model_and_tokenizer
  from src.training_pipeline import load_all_datasets, train_model
  from src.inference_utils import batch_inference, save_predictions

  model, tokenizer = load_model_and_tokenizer()
  train_ds, val_ds, test_ds = load_all_datasets(
      os.path.join(PROJECT_ROOT, "processed_train.json"),
      os.path.join(PROJECT_ROOT, "processed_val.json"),
      os.path.join(PROJECT_ROOT, "processed_test.json"),
  )

  trainer = train_model(
      model=model,
      tokenizer=tokenizer,
      train_dataset=train_ds,
      eval_dataset=val_ds,
      output_dir=os.path.join(PROJECT_ROOT, "results_full_finetune"),
      num_epochs=10,
      batch_size=4,
  )

  trainer.save_model(os.path.join(PROJECT_ROOT, "results_full_finetune/final_model"))

  preds, refs = batch_inference(model, tokenizer, test_ds)
  save_predictions(preds, refs, os.path.join(PROJECT_ROOT, "predictions_full_finetune.json"))

to resume from a checkpoint:

  from src.checkpoint_utils import find_latest_checkpoint

  ckpt = find_latest_checkpoint(os.path.join(PROJECT_ROOT, "results_full_finetune"))
  trainer = train_model(
      model=model, tokenizer=tokenizer,
      train_dataset=train_ds, eval_dataset=val_ds,
      output_dir=os.path.join(PROJECT_ROOT, "results_full_finetune"),
      num_epochs=10, resume_from_checkpoint=ckpt,
  )


for Person 5 - LoRA / PEFT
---------------------------

same path setup, but wrap the model with PEFT first.
use build_trainer() for more control over the training loop:

  import sys, os

  PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
  sys.path.insert(0, PROJECT_ROOT)

  from peft import LoraConfig, get_peft_model, TaskType
  from src.model_loader import load_model_and_tokenizer
  from src.training_pipeline import load_all_datasets, get_training_args, build_trainer
  from src.inference_utils import batch_inference, save_predictions

  model, tokenizer = load_model_and_tokenizer()

  lora_config = LoraConfig(
      task_type=TaskType.SEQ_2_SEQ_LM,
      r=8, lora_alpha=32, lora_dropout=0.1,
  )
  model = get_peft_model(model, lora_config)
  model.print_trainable_parameters()

  train_ds, val_ds, test_ds = load_all_datasets(
      os.path.join(PROJECT_ROOT, "processed_train.json"),
      os.path.join(PROJECT_ROOT, "processed_val.json"),
      os.path.join(PROJECT_ROOT, "processed_test.json"),
  )

  args = get_training_args(
      output_dir=os.path.join(PROJECT_ROOT, "results_lora"),
      num_epochs=10,
  )
  trainer = build_trainer(model, tokenizer, args, train_ds, val_ds)
  trainer.train()

  model.save_pretrained(os.path.join(PROJECT_ROOT, "results_lora/lora_adapters"))

  preds, refs = batch_inference(model, tokenizer, test_ds)
  save_predictions(preds, refs, os.path.join(PROJECT_ROOT, "predictions_lora.json"))


if something breaks
--------------------

check that:
- the json files and dataset_final.csv are in the project root
- the src/ folder has __init__.py
- you're importing from src/ correctly
- you run the script from the project root: python notebooks/03_training_pipeline.py
