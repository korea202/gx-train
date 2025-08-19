import os
PROJECT_ROOT = "/data/ephemeral/home/work/python/gx-train"
DATA_DIR = PROJECT_ROOT + "/data"
CONFIGS_DIR = PROJECT_ROOT + "/configs"
MODELS_DIR = PROJECT_ROOT + "/models"
OUTPUTS_DIR = PROJECT_ROOT + "/outputs"
LOGS_DIR = PROJECT_ROOT + "/logs"

HOUSE_PRICING_DATA = PROJECT_ROOT + "/data/train.csv"

CV_CLS_TRAIN_CSV = DATA_DIR + "/row/train.csv"
CV_CLS_TRAIN_DIR = DATA_DIR + "/row/train"

CV_CLS_TEST_CSV = DATA_DIR + "/row/sample_submission.csv"
CV_CLS_TEST_DIR = DATA_DIR + "/row/test"

CV_CLS_AUGMENT_CSV = DATA_DIR + "/row/augment.csv"
CV_CLS_AUGMENT_DIR = DATA_DIR + "/row/augment"

CV_CLS_MISS_DIR = DATA_DIR + "/row/miss"

NLP_RAW_TRAIN_CSV = DATA_DIR + "/raw/train.csv"
NLP_RAW_DEV_CSV = DATA_DIR + "/raw/dev.csv"
NLP_RAW_TEST_CSV = DATA_DIR + "/raw/test.csv"


IMAGE_SIZE = 380
