#pip install sacrebleu rouge-score
import json
import sys

lora = "../outputs/lora/predictions_lora.json"
full = "../outputs/full_ft/predictions_full_ft.json"

with open(lora, "r", encoding="utf-8") as f:
    lora_data = json.load(f)

with open(full, "r", encoding="utf-8") as f:
    full_data = json.load(f)


# Function to compute Cultural Faithfulness Score (CFS)
# This is a manual evaluation metric where a human reviewer scores
# how culturally appropriate the translation is on a scale of 1–5
def computecfs(predictions, references,inputs):
    scores = []
    for i, (inp, ref, pred) in enumerate(zip(inputs, references, predictions)):
            print(f"English Idiom : {inp}")
            print(f"Reference     : {ref}")
            print(f"Prediction    : {pred}")
            
            # Ask the evaluator to provide a score between 1 and 5
            while True:
                try:
                    score = int(input("Your CFS score (1-5): "))
                    if 1 <= score <= 5:
                        scores.append(score)
                        break
                    else:
                        print("Please enter a number between 1 and 5")
                except ValueError:
                    print("Invalid input, try again")
        
    average = sum(scores) / len(scores)
    # Return both the individual scores and the overall average
    return scores, average