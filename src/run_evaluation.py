#pip install sacrebleu rouge-score
import json
import csv
import os
from bleu_metric import computesacre
from rouge_metric import computerouge
from cultural_metric import computecfs

with open("../outputs/full_ft/predictions_full_ft.json", encoding="utf-8") as f:
    full_data = json.load(f)

with open("../outputs/lora/predictions_lora.json", encoding="utf-8") as f:
    lora_data = json.load(f)

# Extract the original idioms
# The fixed prompt text is removed so we only keep the actual input idiom
full_inputs      = [d["input"].replace("Convert the following English idiom into its natural Urdu equivalent: ", "") for d in full_data["predictions"]]
full_references  = [d["reference"]  for d in full_data["predictions"]]
full_predictions = [d["prediction"] for d in full_data["predictions"]]

lora_inputs      = [d["input"].replace("Convert the following English idiom into its natural Urdu equivalent: ", "") for d in lora_data["predictions"]]
lora_references  = [d["reference"]  for d in lora_data["predictions"]]
lora_predictions = [d["prediction"] for d in lora_data["predictions"]]

full_bleu  = computesacre(full_predictions, full_references)
full_rouge = computerouge(full_predictions, full_references)
lora_bleu  = computesacre(lora_predictions, lora_references)
lora_rouge = computerouge(lora_predictions, lora_references)

# Start manual cultural faithfulness scoring for the Full FT outputs
print("\n Scoring for Fine-Tuning:")
full_cfs_scores, full_cfs_avg = computecfs(full_predictions, full_references, full_inputs)

# Start manual cultural faithfulness scoring for the LoRA outputs
print("\n Scoring for LoRA:")
lora_cfs_scores, lora_cfs_avg = computecfs(lora_predictions, lora_references, lora_inputs)

print("\n RESULTS")
print(f"{'Metric':<15} {'Full FT':>10} {'LoRA':>10}")
print(f"{'BLEU':<15} {full_bleu:>10.2f} {lora_bleu:>10.2f}")
print(f"{'ROUGE-L':<15} {full_rouge:>10.4f} {lora_rouge:>10.4f}")
print(f"{'CFS (avg)':<15} {full_cfs_avg:>10.2f} {lora_cfs_avg:>10.2f}")

os.makedirs("../evaluation_results", exist_ok=True)

# Save the final metric scores into a simple CSV file
with open("../evaluation_results/scores.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Metric", "Full_FT", "LoRA"])
    writer.writerow(["BLEU",    round(full_bleu, 2),   round(lora_bleu, 2)])
    writer.writerow(["ROUGE-L", round(full_rouge, 4),  round(lora_rouge, 4)])
    writer.writerow(["CFS_avg", round(full_cfs_avg, 2), round(lora_cfs_avg, 2)])

with open("../evaluation_results/comparison_table.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Input", "Reference", "Full_FT_Pred", "LoRA_Pred", "Full_FT_CFS", "LoRA_CFS"])
    for i in range(len(full_inputs)):
        writer.writerow([
            full_inputs[i], full_references[i],
            full_predictions[i], lora_predictions[i],
            full_cfs_scores[i], lora_cfs_scores[i]
        ])

print("\nSaved to evaluation_results/")