# Transfer Learning for English Idiom Translation into Urdu

> CSCI316 Big Data Mining and Applications · University of Wollongong in Dubai · Python · HuggingFace Transformers · mT5 · LoRA · Docker · FastAPI

This repository contains an academic NLP experiment that explores whether transfer learning can be used to adapt a multilingual transformer model for English idiom translation into culturally appropriate Urdu equivalents.

The project compares two fine-tuning strategies using the mT5-small model: full fine-tuning and LoRA. The goal was not only to train a model, but also to understand the challenges of applying transfer learning to a low-resource and culturally nuanced language task.

The final models did not achieve usable translation quality. Both approaches received extremely low automatic scores and a Cultural Faithfulness Score of 1.00 out of 5.00. The value of this project lies in the full experimentation pipeline, the comparison between fine-tuning strategies, the custom cultural evaluation method, and the analysis of model failure in a low-resource NLP setting.

---

## Project Overview

Recent advances in natural language processing are driven by large pre-trained models such as BERT and T5, which can be adapted to new tasks through transfer learning. However, these models are mostly trained on high-resource languages, which limits their effectiveness for languages that are less represented in NLP datasets.

Urdu was selected because it is a low-resource language in NLP despite being spoken by millions of people worldwide. Idiom translation was selected because idioms cannot be translated literally. Their meaning depends on cultural and semantic understanding rather than word-level mapping.

For example, an English idiom such as **“Spill the beans”** does not mean spilling actual beans. It means revealing a secret. A successful Urdu translation needs to capture that intended meaning naturally, not translate each word directly.

This project therefore investigates whether a multilingual model can transfer its existing language knowledge to a culturally specific English-to-Urdu idiom translation task.

---

## What This Project Does

- Loads and preprocesses an English-Urdu idiom dataset from an Excel file
- Converts the dataset into CSV format and then into a HuggingFace dataset format
- Cleans the data and removes missing values
- Adds a thematic category field using keyword-based classification
- Groups idioms into 16 thematic categories such as social, health, wisdom, emotions, animals, religion, nature, work, communication, and conflict
- Formats each sample as an instruction-style prompt for sequence-to-sequence translation
- Tokenises the data using the mT5 tokenizer
- Fine-tunes mT5-small using full fine-tuning
- Fine-tunes mT5-small using LoRA through the PEFT library
- Generates predictions for the test set using both trained models
- Evaluates both approaches using BLEU, ROUGE-L, and a manually scored Cultural Faithfulness Score
- Compares the failure patterns of full fine-tuning and LoRA
- Provides a Dockerised FastAPI inference endpoint with Swagger UI

---

## Project Objective

The main objective of this project was to explore whether transfer learning can support English-to-Urdu idiom translation in a low-resource setting.

The project focused on the following questions:

- Can a multilingual model such as mT5-small be adapted to translate English idioms into Urdu?
- Does full fine-tuning perform better than LoRA for this task?
- Can automatic metrics such as BLEU and ROUGE properly evaluate idiom translation?
- What failure patterns appear when fine-tuning a multilingual model on a small idiom dataset?
- What does this reveal about transfer learning for culturally specific NLP tasks?

---

## My Contributions

My specific contributions included:

- **LoRA experimentation** - provided assistance in the PEFT-based LoRA setup, where adapter layers were applied to the mT5-small model while the base model weights remained frozen
- **Evaluation framework** - helped evaluate model outputs using BLEU, ROUGE-L, and the custom Cultural Faithfulness Score
- **Cultural Faithfulness Score review** - manually reviewed predictions against the English idiom and Urdu reference to assess whether the output preserved cultural and idiomatic meaning
- **Docker and API setup** - contributed to the Dockerised setup and FastAPI inference endpoint that allowed predictions to be tested through Swagger UI
- **Documentation and reporting** - co-led the final project report, including the methodology, results, discussion, evaluation, and explanation of model failure

---

## Dataset

The dataset used in this project was `Urdu_Idioms_with_English_Translation.xlsx`, sourced from the HuggingFace dataset uploaded by Ehtisham ul Hassan.

The original dataset contained two columns:

- Urdu Idiom
- English Translation

During preprocessing, the Excel file was converted into CSV format and later transformed into a HuggingFace dataset for training.

| Detail | Value |
| --- | --- |
| Total idiom pairs | 2,122 |
| Training samples | 1,689 |
| Validation samples | 211 |
| Test samples | 212 |
| Split ratio | 80:10:10 |
| Added field | Category |
| Number of thematic categories | 16 |
| Most common category | other |
| Largest named category | social |

---

## Category Distribution

A keyword-based classification approach was used to assign idioms into thematic categories. Initially, more than 1,500 entries were labelled as `other`. After keyword analysis of the English translations, many idioms were reassigned into more meaningful categories. This reduced the `other` category to approximately 700 entries.

| Category | Count |
| --- | ---: |
| other | 701 |
| social | 337 |
| health | 160 |
| wisdom_folly | 127 |
| emotions | 118 |
| body | 95 |
| success | 94 |
| animals | 68 |
| religion_fate | 61 |
| nature | 59 |
| quantity | 59 |
| time_money | 57 |
| work | 40 |
| action | 38 |
| communication | 38 |
| conflict | 35 |
| secrecy | 25 |

---

## Data Challenges

The dataset presented several challenges:

- The dataset was small for a task requiring deep cultural and semantic understanding
- Some English idioms can have more than one valid Urdu equivalent depending on context
- Idioms cannot be translated word by word
- The model needed to learn meaning, not just lexical overlap
- Urdu is underrepresented in many NLP datasets and tools
- The task required cultural mapping between English expressions and Urdu equivalents

These challenges made the project useful for studying the limitations of transfer learning in low-resource NLP.

---

## Methodology

### Preprocessing

The dataset was processed using pandas. Missing values were checked and removed as a precaution. The Excel file was converted into CSV format, and a keyword-based classifier was used to assign each idiom to a thematic category.

Each sample was then converted into an instruction-style prompt. The English idiom was provided as the input, and the Urdu idiom was used as the expected target output.

The text was tokenised using the mT5 tokenizer. Tokenisation converted the text into numerical token IDs that the model could process. The tokenizer also generated attention masks and label sequences required for sequence-to-sequence training.

### Base Model

The base model used in this project was `google/mt5-small`.

mT5-small is a multilingual encoder-decoder transformer model pre-trained on the mC4 multilingual corpus. It was selected because it supports multilingual sequence-to-sequence tasks and is smaller than larger transformer models, making it more practical to fine-tune with limited computational resources.

The encoder processes the English idiom input, while the decoder generates the Urdu output.

### Full Fine-Tuning

Full fine-tuning updates all model parameters during training. In this project, the mT5-small model was fine-tuned on the English-Urdu idiom dataset using HuggingFace Seq2SeqTrainer.

This approach allowed all layers of the model to adapt to the idiom translation task. However, it also required higher computational cost because all model parameters were updated.

After training, the model generated predictions for the 212 test samples. Training logs, model checkpoints, and predictions were saved.

### LoRA

LoRA, or Low-Rank Adaptation, is a parameter-efficient fine-tuning method. Instead of updating the entire model, LoRA adds small low-rank adapter matrices into selected transformer layers while keeping the original pre-trained model weights frozen.

In this project, LoRA adapters were applied to mT5-small using the PEFT library. This reduced the number of trainable parameters and made training more efficient.

The same HuggingFace training pipeline was used, but only the adapter parameters were optimised during training. The adapted model then generated predictions for the test set.

---

## Framework Implementation

The system was implemented using PyTorch and HuggingFace Transformers.

PyTorch handled model training, tensor operations, and GPU acceleration. HuggingFace provided the mT5 model, tokenizer, Seq2SeqTrainer, and training utilities.

The pipeline followed three main stages:

1. Data loading and preprocessing
2. Model training using either full fine-tuning or LoRA
3. Inference and evaluation on the test set

The modular structure allowed the same project pipeline to support both fine-tuning strategies.

---

## Repository Structure

```text
316project2/
├── src/
│   ├── data_preprocessing.py        # Dataset loading, cleaning, prompt formatting, and tokenisation
│   ├── model_loader.py              # mT5-small and tokenizer initialisation
│   ├── train_full_ft.py             # Full fine-tuning training pipeline
│   ├── train_lora.py                # LoRA training pipeline using PEFT
│   ├── inference.py                 # Prediction generation and output saving
│   └── evaluate.py                  # BLEU, ROUGE-L, and CFS evaluation
├── outputs/
│   ├── full_ft/
│   │   ├── checkpoint/              # Full fine-tuning checkpoints
│   │   ├── predictions.json         # Full fine-tuning predictions
│   │   └── training_log.json        # Full fine-tuning training logs
│   └── lora/
│       ├── checkpoint/              # LoRA adapter checkpoints
│       ├── predictions.json         # LoRA predictions
│       └── training_log.json        # LoRA training logs
├── dataset_final.csv                # Final processed dataset
├── app.py                           # FastAPI inference endpoint
├── config.yaml                      # Training configuration
├── Dockerfile                       # Docker image definition
├── requirements.txt                 # Python dependencies
└── README.md
