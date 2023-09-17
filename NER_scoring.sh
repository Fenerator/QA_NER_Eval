
PREDICTION_FILE=NER_ALS_Test_PREDICTIONS.conll
GOLD_FILE=NER_ALS_Test_GOLD.conll #labels

COMBINED_FILE_NAME=NER_ALS_Test_combined.conll

python combine_conll_file_tags.py --predictions $PREDICTION_FILE --labels $GOLD_FILE --output $COMBINED_FILE_NAME

# Scoring
python conlleval.py < $COMBINED_FILE_NAME > "NER_Result_"$PREDICTION_FILE".txt" 
