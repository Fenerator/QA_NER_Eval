import argparse, re
from pathlib import Path
import pandas as pd


def main(args):
    input_file = Path(args.input)
    assert input_file.is_file(), f"Input file {input_file} does not exist"

    gold_file = Path(args.gold)
    print(f"Gold file: {gold_file}")

    output_file = Path(args.output)

    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")

    df = pd.read_csv(input_file, encoding="utf-8", dtype=str, on_bad_lines="skip")
    print(f"Number of lines input: {len(df)}")
    print(f"Colums: {df.columns}")

    df_gold = pd.read_csv(gold_file, encoding="utf-8", dtype=str, on_bad_lines="skip")

    # keep only paragraph, question, prediction (answer), and scores
    if args.submission:
        df_out = df[
            [
                "id",
                "text",
                "question",
                "answer",
                "score_chrf",
                "score_chrf+",
                "score_chrf++",
                "score_rougeL",
                "score_BERTScoreF1",
            ]
        ]
    elif args.gpt:
        df_out = df[
            [
                "id",
                "text",
                "question",
                "gpt-4",
                "score_chrf",
                "score_chrf+",
                "score_chrf++",
                "score_rougeL",
                "score_BERTScoreF1",
            ]
        ]

    elif args.mt0:
        assert len(df) == len(
            df_gold
        ), f"Number of lines in input file {input_file} and gold file {gold_file} do not matchL: len(input)={len(df)}, len(gold)={len(df_gold)}"

        # concatenate the 2 dfs
        df_out = pd.concat([df, df_gold], axis=1)

        df_out = df_out[
            [
                "id",
                "text",
                "question",
                "hyp",
                "score_chrf",
                "score_chrf+",
                "score_chrf++",
                "score_rougeL",
                "score_BERTScoreF1",
            ]
        ]

        df_out.rename({"hyp": "mt0"}, axis=1, inplace=True)

    df_out = df_out.sample(n=args.num_samples)
    print(f"Number of lines output: {len(df_out)}")

    df_out.to_csv(output_file, index=False, encoding="utf-8")

    print(f"Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input csv file containing the baseline predictions",
    )
    parser.add_argument(
        "--gold",
        type=str,
        required=True,
        help="Path to the input csv file which contains the gold answers",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="file path to the output csv file",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        required=True,
        help="Number of samples to take from the input file",
    )
    parser.add_argument("--gpt", action="store_true", help="Process gpt-4 output")
    parser.add_argument("--mt0", action="store_true", help="Process mt0 output")
    parser.add_argument("--submission", action="store_true", help="Process submission")

    args = parser.parse_args()

    main(args)
