#pip install sacrebleu rouge-score
from rouge_score import rouge_scorer
import json
import sys


lora = "../outputs/lora/predictions_lora.json"
full = "../outputs/full_ft/predictions_full_ft.json"

with open(lora, "r", encoding="utf-8") as f:
    lora_data = json.load(f)

with open(full, "r", encoding="utf-8") as f:
    full_data = json.load(f)

# Function to compute the average ROUGE-L score
# ROUGE-L measures similarity between predicted text and reference text
# based on the longest common subsequence between them
def computerouge(predictions, references):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
    rouge_scores = []
    for pred, ref in zip(predictions, references):
        score = scorer.score(ref, pred)
        rouge_scores.append(score['rougeL'].fmeasure)
    return sum(rouge_scores) / len(rouge_scores) 