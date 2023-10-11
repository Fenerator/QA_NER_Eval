
PREDICTION_BASE=/Users/ /dug/Py/QA_NER_Eval/Outputs/Baselines/gpt_4_results/
# PREDICTION_BASE=/Users/dug/Py/QA_NER_Eval/Outputs/Baselines/MT0/
PREDICTION_BASE=/Users/dug/Py/QA_NER_Eval/Outputs/System-CUNI/CUNI-CONTRASTIVE-QA-MRL-submission/

# SYSTEM_NAME=MT0_
SYSTEM_NAME=CContrastive_

GOLD_BASE=/Users/dug/Py/wikiExtract2csv/Data/Test_labels/

# PREDICTION_PREFIX=""
# PREDICTION_PREFIX="zero_shot_"
PREDICTION_PREFIX="CUNI_CONTRASTIVE_QA_"

OUTPUT_PREFIX="sampled_"

array=( AZ.csv ID.csv IG.csv TR.csv UZ.csv YO.csv ALS.csv ) # PREDICTIONS
array2=( QA_AZ_Test.csv QA_ID_Test.csv QA_IG_Test.csv QA_TR_Test.csv QA_UZ_Test.csv QA_YO_Test.csv QA_ALS_Test.csv ) # GOLDS
# array=( AZ.csv  ) # PREDICTIONS
# array2=( QA_AZ_Test.csv  ) # GOLDS
for i in "${!array[@]}"; do
    python sample_paragraphs.py --input $PREDICTION_BASE$PREDICTION_PREFIX${array[i]} --gold $GOLD_BASE${array2[i]} --output $PREDICTION_BASE$SYSTEM_NAME$OUTPUT_PREFIX${array[i]} --num_samples 50 --submission
done
