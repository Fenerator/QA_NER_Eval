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
import random
import sklearn.metrics as metrics
from pathlib import Path

# Local, Debugging mode
debug = False


# Path
if debug:
    BASE_Path = Path("~/Documents/MRL/MC_QA_competition/MC_competition")
    
    input_dir = BASE_Path / 'final_phase'  # Input from ingestion program
    os.makedirs(input_dir / 'output', exist_ok=True)
    output_dir = input_dir / "output"  # To write the scores
    
    reference_dir = input_dir / "reference_data"  # Ground truth data
    prediction_dir = '~/Downloads/perfect_mc_submission'  # Prediction made by the model
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
    return ["MC_AZ", "MC_YO", "MC_TR", "MC_IG", "MC_ALS"]

def csv_to_list(csv_file, column_name):
    """Convert a column of a csv file to a list."""
    print(f'Reading file: {csv_file}')
    df = pd.read_csv(csv_file, encoding="utf-8")
    if debug:
        print(df.columns)
    return list(df[column_name])
    
def get_data(dataset):
    """Get ground truth (y_test) and predictions (y_pred) from the dataset name."""
    y_test = csv_to_list(os.path.join(reference_dir, dataset + "_reference_test.csv"), 'label')
    y_pred = csv_to_list(os.path.join(prediction_dir, dataset + "_test.predict"), 'prediction')
    
    assert len(y_test) == len(y_pred), f'Length of y_test ({len(y_test)}) and y_pred ({len(y_pred)}) are not equal.'
    return y_test, y_pred


def print_bar():
    """Display a bar ('----------')"""
    print("-" * 10)


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
        if len(y_pred) == 0:
            accuracy, f1, precision, recall = 0.0, 0.0, 0.0, 0.0
        else:
            accuracy = metrics.accuracy_score(y_test, y_pred, normalize=True)
            f1 = metrics.f1_score(y_test, y_pred, average='micro')
            precision = metrics.precision_score(y_test, y_pred, average='micro')
            recall = metrics.recall_score(y_test, y_pred, average='micro')

        sub_scores["accuracy"] = accuracy
        sub_scores["f1"] = f1 
        sub_scores["precision"] = precision
        sub_scores["recall"] = recall
    
        write_file(html_file, f"<p>accuracy:  {accuracy:.3f}</p>")
        write_file(html_file, f"<p>f1: {f1:.3f}</p>")
        write_file(html_file, f"<p>precision: {precision:.3f}</p>")
        write_file(html_file, f"<p>recall: {recall:.3f}</p>")
        write_file(html_file, "<hr>")
        
        detailed_scores[dataset] = sub_scores
    
    # calculate weighted mean over all datasets and metrics    
    print_bar()
    print(f'Calculating weighted mean over languages ...')
    scores = {'accuracy': [], 'f1': [], 'precision': [], 'recall': []}    
    for dataset in get_dataset_names():
        for metric in detailed_scores[dataset]:
            scores[metric].append(detailed_scores[dataset][metric])
    
    weights = np.array(lens) / sum(lens)
    assert np.isclose(sum(weights), 1.0), f'Sum of weights is not 1.0: {sum(weights)}'
        
    mean_scores = {}
    for metric in scores:
        mean_scores[f'mean_{metric}'] = np.average(scores[metric], weights=weights)     
    
    write_file(html_file, f"<h1>Final Scores (averaged over languages)</h1>")
    write_file(html_file, f"<p>accuracy:  {mean_scores['mean_accuracy']:.3f}</p>")
    write_file(html_file, f"<p>f1: {mean_scores['mean_f1']:.3f}</p>")
    write_file(html_file, f"<p>precision: {mean_scores['mean_precision']:.3f}</p>")
    write_file(html_file, f"<p>recall: {mean_scores['mean_recall']:.3f}</p>")
    write_file(html_file, "<hr>")
    
    # Write scores for leaderboard (accuracy only)
    leaderboard_scores = {}
    leaderboard_scores["mean_accuracy"] = mean_scores['mean_accuracy']
    
    for dataset in get_dataset_names():
        leaderboard_scores[dataset] = detailed_scores[dataset]["accuracy"]
    print_bar()
    print("Scoring program finished. Writing scores...")
    
    if debug:
        print(detailed_scores)
        print(leaderboard_scores)
    write_file(score_file, json.dumps(leaderboard_scores))



if __name__ == "__main__":
    main()
