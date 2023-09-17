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
