import argparse
from pathlib import Path
import pandas as pd
import evaluate


def read_csv(file_path):
    df = pd.read_csv(file_path, encoding="utf-8", dtype=str, on_bad_lines="skip")
    print(f"Number of tasks: {len(df)}")

    questions = df["question"].tolist()
    answers = df["answer"].tolist()
    ids = df["id"].tolist()

    assert (
        len(questions) == len(answers) == len(ids)
    ), f"Number of questions, answers and ids does not match in file {file_path}"

    return questions, answers, ids


def main(args):
    prediction_file = Path(args.predictions)
    assert (
        prediction_file.is_file()
    ), f"Predictions file {prediction_file} does not exist"

    label_file = Path(args.labels)
    assert label_file.is_file(), f"Labels file {label_file} does not exist"

    print(f"Prediction file: {prediction_file}")
    print(f"Label file: {label_file}")

    _, predictions_a, predictions_ids = read_csv(prediction_file)
    _, labels_a, labels_ids = read_csv(prediction_file)

    for i in range(len(predictions_ids)):
        assert predictions_ids[i] == labels_ids[i], f"IDs do not match at index {i}"

    # calculate ChrF, there should be one reference sub-list for each prediction sentence.
    chrf = evaluate.load("chrf")
    results = chrf.compute(predictions=predictions_a, references=labels_a)
    print(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="Path to the input csv file containing the predictions (answers)",
    )
    parser.add_argument(
        "--labels",
        type=str,
        required=True,
        default="article_question_pairs.csv",
        help="Path to the input csv file containing the true labels (answers)",
    )

    args = parser.parse_args()

    main(args)
