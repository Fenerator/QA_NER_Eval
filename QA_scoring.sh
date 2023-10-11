
# PREDICTION_BASE=/Users/dug/Py/QA_NER_Eval/Outputs/Baselines/MT0/
PREDICTION_BASE=/Users/dug/Py/QA_NER_Eval/Outputs/Baselines/gpt_4_results/
GOLD_BASE=/Users/dug/Py/wikiExtract2csv/Data/Test_labels/ #labels

PREDICTION_PREFIX=""

FILE=QA_AZ_Test.csv
PREDICTION=AZ.csv
# # python evaluate_QA.py --predictions $PREDICTION_BASE$PREDICTION_PREFIX$PREDICTION --labels $GOLD_BASE$FILE --results $PREDICTION_BASE$PREDICTION_PREFIX"Evaluation.txt" --baseline
python evaluate_QA.py --predictions $PREDICTION_BASE$PREDICTION_PREFIX$PREDICTION --labels $GOLD_BASE$FILE --results $PREDICTION_BASE$PREDICTION_PREFIX"Evaluation.txt" --gpt

# FILE=QA_ID_Test.csv
# PREDICTION=ID.csv
# # python evaluate_QA.py --predictions $PREDICTION_BASE$PREDICTION_PREFIX$PREDICTION --labels $GOLD_BASE$FILE --results $PREDICTION_BASE$PREDICTION_PREFIX"Evaluation.txt" --baseline
# python evaluate_QA.py --predictions $PREDICTION_BASE$PREDICTION_PREFIX$PREDICTION --labels $GOLD_BASE$FILE --results $PREDICTION_BASE$PREDICTION_PREFIX"Evaluation.txt" --gpt

# FILE=QA_IG_Test.csv
# PREDICTION=IG.csv
# # python evaluate_QA.py --predictions $PREDICTION_BASE$PREDICTION_PREFIX$PREDICTION --labels $GOLD_BASE$FILE --results $PREDICTION_BASE$PREDICTION_PREFIX"Evaluation.txt" --baseline
# python evaluate_QA.py --predictions $PREDICTION_BASE$PREDICTION_PREFIX$PREDICTION --labels $GOLD_BASE$FILE --results $PREDICTION_BASE$PREDICTION_PREFIX"Evaluation.txt" --gpt

# FILE=QA_TR_Test.csv
# PREDICTION=TR.csv
# # python evaluate_QA.py --predictions $PREDICTION_BASE$PREDICTION_PREFIX$PREDICTION --labels $GOLD_BASE$FILE --results $PREDICTION_BASE$PREDICTION_PREFIX"Evaluation.txt" --baseline
# python evaluate_QA.py --predictions $PREDICTION_BASE$PREDICTION_PREFIX$PREDICTION --labels $GOLD_BASE$FILE --results $PREDICTION_BASE$PREDICTION_PREFIX"Evaluation.txt" --gpt

# FILE=QA_UZ_Test.csv
# PREDICTION=UZ.csv
# # python evaluate_QA.py --predictions $PREDICTION_BASE$PREDICTION_PREFIX$PREDICTION --labels $GOLD_BASE$FILE --results $PREDICTION_BASE$PREDICTION_PREFIX"Evaluation.txt" --baseline
# python evaluate_QA.py --predictions $PREDICTION_BASE$PREDICTION_PREFIX$PREDICTION --labels $GOLD_BASE$FILE --results $PREDICTION_BASE$PREDICTION_PREFIX"Evaluation.txt" --gpt

# FILE=QA_YO_Test.csv
# PREDICTION=YO.csv
# # python evaluate_QA.py --predictions $PREDICTION_BASE$PREDICTION_PREFIX$PREDICTION --labels $GOLD_BASE$FILE --results $PREDICTION_BASE$PREDICTION_PREFIX"Evaluation.txt" --baseline
# python evaluate_QA.py --predictions $PREDICTION_BASE$PREDICTION_PREFIX$PREDICTION --labels $GOLD_BASE$FILE --results $PREDICTION_BASE$PREDICTION_PREFIX"Evaluation.txt" --gpt

FILE=QA_ALS_Test.csv
PREDICTION=ALS.csv
# python evaluate_QA.py --predictions $PREDICTION_BASE$PREDICTION_PREFIX$PREDICTION --labels $GOLD_BASE$FILE --results $PREDICTION_BASE$PREDICTION_PREFIX"Evaluation.txt" --baseline
# python evaluate_QA.py --predictions $PREDICTION_BASE$PREDICTION_PREFIX$PREDICTION --labels $GOLD_BASE$FILE --results $PREDICTION_BASE$PREDICTION_PREFIX"Evaluation.txt" --gpt
