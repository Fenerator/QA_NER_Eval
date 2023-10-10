
SUBMISSION=CUNI-NER-MRL-submission
PREDICTION_BASE="/Users/dug/Py/QA_NER_Eval/Outputs/System-CUNI/"$SUBMISSION"/"
GOLD_BASE=/Users/dug/Py/wikiExtract2csv/Data/NER_normalized/ #labels

PREDICTION_PREFIX=CUNI_

for FILE in NER_ALS_Test.conll NER_AZ_Test.conll NER_IG_Test.conll NER_TR_Test.conll NER_YO_Test.conll; do
    COMBINED_FILE=$PREDICTION_BASE$PREDICTION_PREFIX$FILE"_combined.conll"
    echo $SUBMISSION"/"$PREDICTION_PREFIX$FILE"" >> $PREDICTION_BASE$PREDICTION_PREFIX"NER_Evaluation.txt"
    # remove potential data tags from the gold standard
    python ../wikiExtract2csv/NER_postprocessing.py --input $GOLD_BASE$FILE  --output $GOLD_BASE$FILE     
    python combine_conll_file_tags.py --predictions $PREDICTION_BASE$PREDICTION_PREFIX$FILE --labels $GOLD_BASE$FILE --output $COMBINED_FILE
    python conlleval.py < $COMBINED_FILE >> $PREDICTION_BASE$PREDICTION_PREFIX"NER_Evaluation.txt" 
    echo "==============\n" >> $PREDICTION_BASE$PREDICTION_PREFIX"NER_Evaluation.txt" 
done
