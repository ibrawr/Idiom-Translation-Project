#pip install sacrebleu rouge-score
import json
import sys
import sacrebleu

lora = "../outputs/lora/predictions_lora.json"
full = "../outputs/full_ft/predictions_full_ft.json"

## Load LoRA predictions JSON file
with open(lora, "r", encoding="utf-8") as f:
    lora_data = json.load(f)

# Load Full Fine-Tuning predictions JSON file
with open(full, "r", encoding="utf-8") as f:
    full_data = json.load(f)

# Function to compute BLEU score using SacreBLEU
def computesacre(predictions, references):
    bleu = sacrebleu.corpus_bleu(predictions, [references])
    return bleu.score