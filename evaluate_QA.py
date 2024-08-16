# The scoring program compute scores from:
# - The ground truth
# - The predictions made by the candidate model

# Imports
import json
import os
import io
import base64
import numpy as np
import pandas as pd

# from sklearn.metrics import accuracy_score
import evaluate

# Local, Debugging mode
debug = False

from transformers import logging
logging.set_verbosity_error()

# Path
if debug:
    input_dir = "/Users/dug/Documents/CodaBench/mini-automl_for_data_only/bundle_V4_one_phase_full/final_phase"  # Input from ingestion program
    os.makedirs('/Users/dug/Documents/CodaBench/mini-automl_for_data_only/bundle_V4_one_phase_full/final_phase/output', exist_ok=True)
    output_dir = "/Users/dug/Documents/CodaBench/mini-automl_for_data_only/bundle_V4_one_phase_full/final_phase/output/"  # To write the scores
    reference_dir = os.path.join(input_dir, "reference_data")  # Ground truth data
    prediction_dir = '/Users/dug/Documents/CodaBench/mini-automl_for_data_only/submission_V4_full'  # Prediction made by the model
else:
    input_dir = "/app/input"  # Input from ingestion program
    output_dir = "/app/output/"  # To write the scores
    reference_dir = os.path.join(input_dir, "ref")  # Ground truth data
    prediction_dir = os.path.join(input_dir, "res")  # Prediction made by the model
    
score_file = os.path.join(output_dir, "scores.json")  # Scores
html_file = os.path.join(output_dir, "detailed_results.html")  # Detailed feedback


def write_file(file, content):
    """Write content in file."""
    with open(file, "a", encoding="utf-8") as f:
        f.write(content)


def get_dataset_names():
    """Return the names of the datasets."""
    return ["QA_AZ", "QA_YO", "QA_TR", "QA_IG", "QA_ALS"]

def csv_to_list(csv_file, column_name):
    """Convert a column of a csv file to a list."""
    print(f'Reading file: {csv_file}')
    df = pd.read_csv(csv_file, encoding="utf-8")
    if debug:
        print(df.columns)
    return list(df[column_name])
    
def get_data(dataset):
    """Get ground truth (y_test) and predictions (y_pred) from the dataset name."""
    y_test = csv_to_list(os.path.join(reference_dir, dataset + "_reference_test.csv"), 'answer')
    y_pred = csv_to_list(os.path.join(prediction_dir, dataset + "_test.predict"), 'prediction')
    
    assert len(y_test) == len(y_pred), f'Length of y_test ({len(y_test)}) and y_pred ({len(y_pred)}) are not equal.'
    return y_test, y_pred


def print_bar():
    """Display a bar ('----------')"""
    print("-" * 10)


def calculate_chrf(predictions, labels):
    """calculate ChrF, and normalize, there should be one reference sub-list for each prediction sentence."""
    print(f'Calculating ChrF ...')
    chrf = evaluate.load("chrf")

    charf = [
        chrf.compute(predictions=[prediction], references=[[label]], word_order=0)[
            "score"
        ]
        for prediction, label in zip(predictions, labels)
    ]
    
    charf = sum(charf) / len(charf)
    
    

    charf1 = [
        chrf.compute(predictions=[prediction], references=[[label]], word_order=1)[
            "score"
        ]
        for prediction, label in zip(predictions, labels)
    ]
    
    charf1 = sum(charf1) / len(charf1)
    
    charf2 = [
        chrf.compute(predictions=[prediction], references=[[label]], word_order=2)[
            "score"
        ]
        for prediction, label in zip(predictions, labels)
    ]
    
    charf2 = sum(charf2) / len(charf2) 
    
    # normalize
    return charf / 100, charf1 / 100, charf2 / 100


def calculate_rougeL(predictions, labels):
    print(f'Calculating RougeL ...')
    rouge = evaluate.load("rouge")

    rougeL = [
        rouge.compute(predictions=[prediction], references=[[label]])["rougeL"]
        for prediction, label in zip(predictions, labels)
    ]
    
    return sum(rougeL) / len(rougeL)


def calculate_BERTScore(predictions, labels):
    print(f'Calculating BERTScore ...')
    bertscore = evaluate.load("bertscore")
    results = [
        bertscore.compute(
            predictions=[prediction], references=[label], model_type="roberta-base"
        )["f1"].pop()
        for prediction, label in zip(predictions, labels)
    ]

    return sum(results) / len(results)

def main():
    """The scoring program."""
    print_bar()
    print("Scoring program.")
    # Initialized detailed results
    write_file(
        html_file, "<h1>Detailed Results</h1>"
    )  # Create the file to give real-time feedback
    detailed_scores = {} # [dataset][metric]
    lens = []
    for dataset in get_dataset_names():  # Loop over datasets
        sub_scores = {}
        print_bar()
        print(f'Dataset: {dataset}')
        write_file(html_file, f"<h2>Dataset: {dataset}</h2>")
        
        # Read data
        y_test, y_pred = get_data(dataset)
        lens.append(len(y_test))
        
        # Compute scores
        # normalized chrf scores
        chrf_result, chrf1_result, chrf2_result = calculate_chrf(
        y_pred, y_test
        )
        rougeL_result = calculate_rougeL(y_pred, y_test)
        bertScore_result = calculate_BERTScore(y_pred, y_test)
        
        sub_scores["chrf"] = chrf_result
        sub_scores["chrf1"] = chrf1_result 
        sub_scores["chrf2"] = chrf2_result
        sub_scores["rougeL"] = rougeL_result
        sub_scores["bertScore"] = bertScore_result
        sub_scores["mean"] = np.mean([chrf_result, chrf1_result, chrf2_result, rougeL_result, bertScore_result])
        
        write_file(html_file, f"<p>ChrF:  {chrf_result:.3f}</p>")
        write_file(html_file, f"<p>ChrF1: {chrf1_result:.3f}</p>")
        write_file(html_file, f"<p>ChrF2: {chrf2_result:.3f}</p>")
        write_file(html_file, f"<p>RougeL: {rougeL_result:.3f}</p>")
        write_file(html_file, f"<p>BERTScore: {bertScore_result:.3f}</p>")
        write_file(html_file, "<hr>")
        write_file(html_file, f"<p>Mean ({dataset}): {sub_scores['mean']:.3f}</p>")
        write_file(html_file, "<hr>")
        
        detailed_scores[dataset] = sub_scores
    
    # calculate weighted mean over all datasets and metrics
    scores = {}    
    for dataset in get_dataset_names():
        scores[dataset] = detailed_scores[dataset]["mean"]
    
    means = [scores[dataset] for dataset in get_dataset_names()]     
    weights = np.array(lens) / sum(lens)
    assert np.isclose(sum(weights), 1.0), f'Sum of weights is not 1.0: {sum(weights)}'

    scores["weighted_mean"] = np.average(means, weights=weights)
    write_file(html_file, f"<h1>Final Score</h1>")
    write_file(html_file, "<hr>")
    write_file(html_file, f"<p>Weighted Mean: {scores['weighted_mean']:.3f}</p>")
    write_file(html_file, "<hr>")
    
    # Write scores
    print_bar()
    print("Scoring program finished. Writing scores.")
    print(f'Detailed Scores:\n{detailed_scores}')
    write_file(score_file, json.dumps(scores))



if __name__ == "__main__":
    main()
