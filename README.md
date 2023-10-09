# Evaluation of NER and QA

## NER Evaluation

using conlleval-python from [sighsmile/conlleval](https://github.com/sighsmile/conlleval).

1. Clone the [sighsmile/conlleval](https://github.com/sighsmile/conlleval) repository. And move the `conlleval.py` file to the current directory.
2. Append the predictions to the conll file such that each line has the following format:

    ```token true_label predicted_label``` using the following command:

    ```bash
    python combine_conll_file_tags.py --predictions NER_ALS_Test_PREDICTIONS.conll --labels NER_ALS_Test_GOLD.conll --output NER_ALS_Test_combined.conll
    ```

3. Score the `conll` file using:

    ```bash
    python conlleval.py < NER_ALS_Test_combined.conll > NER_ALS_Test_Result.txt     
    ```

4. alternatively use the NER_scoring.sh script to do all the above steps:

    ```bash
    ./NER_scoring.sh
    ```

## QA Evaluation

### Metrics

All metrics are calculated using implementations from [Huggingface Evaluate Metric](https://huggingface.co/evaluate-metric). The following metrics are used:

- ChrF ('char_order': 6, 'word_order': 0, 'beta': 2)
- ChrF+ ('char_order': 6, 'word_order': 1, 'beta': 2)
- ChrF++ ('char_order': 6, 'word_order': 2, 'beta': 2)
- RougeL
- BERTScore F1 using embeddings from RobertaBase ('hashcode': 'roberta-base_L10_no-idf_version=0.3.12(hug_trans=4.34.0)

### Usage

Use `QA_scoring.sh` to see an example to score all predictions of a system. The underlying script is `evaluate_QA.py` which can be used as follows:

```bash
python evaluate_QA.py --predictions PRED_FILE --labels GOLD_FILE --results RESULTS_FILE
```

- `GOLD_FILE` is the file containing the gold answers
- `PRED_FILE` is the file containing a system's predicted answer
- `OUTPUT_FILE` is the file to which the average scores per language will be written.
  
Detailed scores (the scores for each paragraph) will be appended as columns to the `PRED_FILE`.
