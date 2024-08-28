from enum import Enum

# Current implementation deflates the number of pose to match the number of move points
# Another implementation may be to inflate the number of move sequences to match number of pose sequences
encoder_type = Enum('EncoderType', ['LINEAR_TRANSFORMER', 'LSTM_TRANSFORMER', 'LSTM'])

# General configuration
SEQUENCE_LENGTH = 6
EMBEDDING_DIM = 16
RANDOM_SEED = 42
POSE_N_FEATURES = 18
MOVE_N_FEATURES = 6
TEST_SIZE = 0.15
LEARNING_RATE = 1e-4
DEFLATION_FACTOR = 18 # Video frames to movement frames ratio is 18:1 
THRESHOLD = 3

# Encoder Hyper-parameters
ENCODER_TYPE = encoder_type.LINEAR_TRANSFORMER
ENCODER_LEARNING_RATE = 1e-4
ENCODER_EPOCHS = 10
ENCODER_BATCH_SIZE = 50
TAU = 5
LAM = 2
BALANCE_ENCODER_DATA = True
SAVE_MODEL_PATH = "./models/sq6ed16th3_linear_transformer_e10b50t5l2bt_h8l2_encoder"
# Transformer Configs
# NOTE: This number modifies the the classifier and encoder model 
#       When loading encoder for classifier, ensure that these values match the loaded model
N_HEAD = 8
N_LAYER = 2

# Classifier Hyper-parameters 
CLASSIFIER_ENCODER_TYPE = encoder_type.LINEAR_TRANSFORMER
CLASSIFIER_LEARNING_RATE = 1e-4
BALANCE_CLASSIFIER_DATA = True
CLASSIFIER_EPOCHS = 10
CLASSIFIER_BATCH_SIZE = 50
FREEZE_ENCODER_STATE = True
# NOTE: Loaded model needs to be match defined CLASSIFIER_ENCODER_TYPE
LOAD_MODEL_PATH = "./models/sq6ed16th3_linear_transformer_e10b50t5l2bt_h8l2_encoder"