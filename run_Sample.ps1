$BUCKET_PATH = "gs://{your bucket name}"
$TRAIN_DATA = $BUCKET_PATH + "/data/Arithmetic_pad_x.csv"
$TEST_DATA = $BUCKET_PATH + "/data/Arithmetic_y.csv"
$NOW_TIME = Get-Date -UFormat "%Y_%m_%d_%H_%M"
$JOB_NAME = "training_" + $NOW_TIME
$OUTPUT_PATH = $BUCKET_PATH + "/" + $JOB_NAME
$INPUT_MODEL_PATH = $BUCKET_PATH + "/training_2000_00_00_00_00/classify_20.pt"
$REGION = "{your region}"

gcloud ml-engine jobs submit training $JOB_NAME `
--runtime-version 1.10 --python-version 3.5 `
--module-name trainer.task --package-path trainer/ `
--region $REGION --job-dir $OUTPUT_PATH --config config.yaml `
-- `
--train-data $TRAIN_DATA --test-data $TEST_DATA `
--output-model $OUTPUT_PATH --epochs 5 `
# --input-model $INPUT_MODEL_PATH
