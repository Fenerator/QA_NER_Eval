
PREDICTION_BASE=/Users/dug/Py/QA_NER_Eval/Outputs/System-CUNI/CUNI-QA-MRL-submission/
GOLD_BASE=/Users/dug/Py/wikiExtract2csv/Data/Test_labels/ #labels

PREDICTION_PREFIX=CUNI_

FILE=QA_AZ_Test.csv
python evaluate_QA.py --predictions $PREDICTION_BASE$PREDICTION_PREFIX$FILE --labels $GOLD_BASE$FILE --results $PREDICTION_BASE$PREDICTION_PREFIX"Evaluation.txt"

