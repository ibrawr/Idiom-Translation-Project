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
```
---

## Evaluation Metrics

Three evaluation methods were used.

### BLEU

BLEU measures n-gram precision between the predicted translation and the reference translation. It is commonly used for machine translation, but it is limited for idiom translation because a correct idiom may use different wording from the reference.

### ROUGE-L

ROUGE-L measures the longest common subsequence between the prediction and the reference. Like BLEU, it relies on surface-level overlap and may not capture cultural meaning.

### Cultural Faithfulness Score

The Cultural Faithfulness Score was designed for this project to evaluate whether the predicted Urdu output preserved the idiomatic meaning of the English source.

It used a 5-point scale:

| Score | Meaning |
| --- | --- |
| 5 | Perfect idiomatic translation |
| 4 | Culturally natural |
| 3 | Understandable but not idiomatic |
| 2 | Partially correct |
| 1 | Literal or incorrect translation |

CFS scoring was performed manually by reviewing each prediction alongside the English idiom and the Urdu reference.

---

## Quantitative Results

Both models performed poorly across all evaluation metrics.

| Metric | Full Fine-Tuning | LoRA | Notes |
| --- | ---: | ---: | --- |
| BLEU | 0.17 | 0.03 | Both near zero |
| ROUGE-L | 0.0000 | 0.0000 | Zero lexical overlap with references |
| CFS Average | 1.00 / 5.00 | 1.00 / 5.00 | All predictions were incorrect |

Full fine-tuning achieved a higher BLEU score than LoRA, but both scores were still extremely low. ROUGE-L was 0.0000 for both models, showing no meaningful lexical overlap with the reference translations.

The Cultural Faithfulness Score was 1.00 out of 5.00 for both models, meaning neither approach captured idiomatic or cultural meaning.

---

## Cultural Evaluation Results

CFS scoring was applied to all 212 test predictions for both models.

| Score | Meaning | Full FT % | LoRA % |
| --- | --- | ---: | ---: |
| 5 | Perfect idiomatic translation | 0% | 0% |
| 4 | Culturally natural | 0% | 0% |
| 3 | Understandable but not idiomatic | 0% | 0% |
| 2 | Partially correct | 0% | 0% |
| 1 | Literal or incorrect translation | 100% | 100% |

| Model | Average CFS |
| --- | ---: |
| Full Fine-Tuning | 1.00 / 5.00 |
| LoRA | 1.00 / 5.00 |

Neither model captured the idiomatic intent of the English expressions in natural Urdu.

---

## Qualitative Examples

| Input Idiom | Reference Meaning | Full Fine-Tuning Output Pattern | LoRA Output Pattern |
| --- | --- | --- | --- |
| A CHIP OF THE OLD BLOCK | Urdu equivalent for resembling a parent | Repeated short phrase | Long incoherent Urdu string |
| A friend in need is a friend indeed | Urdu equivalent for a true friend helping in difficulty | Repeated unrelated phrase | Garbled Urdu text |
| Break the ice | Urdu equivalent for starting conversation | Repeated unrelated phrase | Garbled Urdu text |
| BURST INTO FLAME | Urdu equivalent for catching fire | Repeated short phrase | Incoherent output |
| Time waits for no one | Urdu equivalent for time not waiting for anyone | Repeated short phrase | Incoherent output |

The full fine-tuning model repeatedly produced a small set of short Urdu phrases regardless of the input idiom. This suggests model collapse, where the model learns a safe repeated response instead of learning task-specific mappings.

The LoRA model produced longer but incoherent Urdu outputs. This suggests that the adapter layers failed to meaningfully redirect the base model toward the idiom translation task.

---

## Method Comparison

| Attribute | Full Fine-Tuning | LoRA |
| --- | --- | --- |
| Translation quality | Collapsed into 3 to 4 repeated short phrases | Generated long incoherent garbled text |
| Fluency | Short but meaningless | Long but unreadable |
| Consistency | Highly consistent, same wrong answer repeatedly | Inconsistent and different garbled output each time |
| Training efficiency | Higher compute, all model parameters updated | Lower compute, only adapter parameters trained |
| BLEU score | 0.17 | 0.03 |
| CFS average | 1.00 / 5.00 | 1.00 / 5.00 |

Full fine-tuning had more capacity because it updated all model parameters, but it still collapsed into repetitive output. LoRA was more efficient but did not have enough flexibility to learn the task effectively with the limited dataset.

---

## Discussion

### Full Fine-Tuning

Full fine-tuning showed that some general language knowledge transferred from the pre-trained mT5-small model. The outputs were often short and readable, which suggests that the model retained some ability to generate Urdu text.

However, the model failed to translate idioms correctly. It often generated generic Urdu phrases unrelated to the input idiom. It also repeated similar outputs across many test samples.

This likely happened because the dataset was too small for the complexity of idiom translation. With only around 1,689 training samples, the model may have overfitted and relied on high-probability Urdu patterns rather than learning accurate idiomatic mappings.

### LoRA

LoRA showed that the model could be trained more efficiently by updating far fewer parameters. This made it computationally cheaper than full fine-tuning.

However, the outputs were less stable than full fine-tuning. Many predictions were fragmented, garbled, or unreadable. Because the base model weights remained frozen and only small adapter matrices were trained, the model had limited flexibility to adapt to the task.

The combination of a small dataset, a compact base model, and the difficulty of idiom translation made LoRA ineffective for this project.

---

## Key Findings

- Transfer learning alone was not enough for English-to-Urdu idiom translation in this setting
- Full fine-tuning performed slightly better than LoRA numerically, but still failed in practice
- LoRA was more computationally efficient but produced less readable outputs
- Both models failed to capture cultural and idiomatic meaning
- BLEU and ROUGE were not sufficient for evaluating idiom translation quality
- Manual cultural evaluation was necessary to understand the real quality of the predictions
- The small dataset size was a major limitation
- Idiom translation requires cultural and semantic understanding, not just multilingual pre-training

---

## Project Status

This project is complete as an academic NLP experiment for CSCI316 Big Data Mining and Applications.

The final models did not achieve usable translation quality. Both full fine-tuning and LoRA received a Cultural Faithfulness Score of 1.00 out of 5.00. The project is included as a research-style portfolio project because it demonstrates the full NLP experimentation process, including data preparation, transfer learning, parameter-efficient fine-tuning, evaluation, Dockerised deployment, and failure analysis.

---

## Future Work

Future improvements could include:

- Expanding the dataset with more English-Urdu idiom pairs
- Using larger multilingual models with stronger translation capabilities
- Applying data augmentation to increase training diversity
- Using prompt-based learning or instruction-tuned multilingual models
- Incorporating external Urdu linguistic resources
- Improving the category classification method beyond keyword matching
- Testing models designed specifically for Urdu or South Asian languages
- Using human evaluation from native Urdu speakers for more reliable cultural assessment

---

## Getting Started

### Prerequisites

Make sure Docker Desktop is installed and running.

### Running with Docker

Open a terminal in the project root folder. The folder should contain:

- `Dockerfile`
- `app.py`
- `config.yaml`
- `requirements.txt`
- `src/`
- `outputs/`
- `dataset_final.csv`

Build the Docker image:

```bash
docker build --no-cache -t idioms-316project2:latest .
```

Run the container:

```powershell
docker run -p 8001:8000 -v "$($PWD.Path)\outputs:/app/outputs" -v "$($PWD.Path)\dataset_final.csv:/app/dataset_final.csv" idioms-316project2:latest
```

Open the API documentation in your browser:

```text
http://127.0.0.1:8001/docs
```

The container runs on port 8000 internally and is exposed on host port 8001 to avoid conflicts with other local services.

---

## Testing Inference

Use the `POST /predict` endpoint in Swagger UI.

Example request:

```json
{
  "idiom": "Empty vessels make more noise.",
  "model_type": "full_ft"
}
```

The API uses the mounted `outputs/` directory and `dataset_final.csv` file so the container can access model outputs and fallback data at runtime.

---

## Running Locally

```bash
pip install -r requirements.txt
python app.py
```

---

## Dependencies

```text
torch
transformers
peft
datasets
evaluate
sacrebleu
rouge-score
pandas
openpyxl
fastapi
uvicorn
```

---

## Tech Stack

**Language:** Python 3.11  
**Model:** mT5-small, `google/mt5-small`  
**Training Framework:** PyTorch, HuggingFace Transformers, Seq2SeqTrainer  
**Parameter-Efficient Fine-Tuning:** LoRA using PEFT  
**Evaluation:** BLEU, ROUGE-L, Cultural Faithfulness Score  
**Data Processing:** pandas, HuggingFace Datasets  
**Inference API:** FastAPI, Uvicorn  
**Containerisation:** Docker  
**Dataset Source:** HuggingFace dataset by Ehtisham ul Hassan  

---

## Skills Demonstrated

- Transfer learning for multilingual NLP
- Sequence-to-sequence model fine-tuning
- HuggingFace Transformers and Seq2SeqTrainer
- Parameter-efficient fine-tuning with LoRA
- PEFT library implementation
- mT5-small model training and inference
- Dataset cleaning and preprocessing with pandas
- Instruction-style prompt formatting
- Tokenisation, attention masks, and label preparation
- Manual evaluation of culturally nuanced translation outputs
- Custom Cultural Faithfulness Score design
- BLEU and ROUGE-L evaluation
- Model failure diagnosis and collapse analysis
- Docker containerisation
- FastAPI endpoint development
- Swagger UI testing
- Low-resource language experimentation

---

## References

Ehtisham ul Hassan. (2024). Urdu Idioms with English Translation dataset. HuggingFace.

Hassan, M. T., Ahmed, J., and Awais, M. (2026). Qalb: Largest State-of-the-Art Urdu Large Language Model for 230M Speakers with Systematic Continued Pre-training. arXiv.

---

## Author

**Ibrar Bhatti**  
GitHub: [ibrawr](https://github.com/ibrawr)

