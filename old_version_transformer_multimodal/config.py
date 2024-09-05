# Current implementation deflates the number of pose to match the number of move points
# Another implementation may be to inflate the number of move sequences to match number of pose sequences
import random

MAX_SEED = 4294967295
seed = random.randrange(MAX_SEED)

# General Configuration
IS_RANDOM = True
if IS_RANDOM:
    print(f"Random seed: {seed}")
    RANDOM_SEED = seed
else:
    RANDOM_SEED = 42

BATCH_SIZE = 32
SEQUENCE_LENGTH = 6
EMBEDDING_DIM = 16
N_HEAD = 8 # Transformer layer
N_LAYERS = 2 # Transformer layer
POSE_N_FEATURES = 18
MOVE_N_FEATURES = 6
TEST_SIZE = 0.15
VALIDATION_SIZE = 0.10
LEARNING_RATE = 1e-4
DEFLATION_FACTOR = 18 # Video frames to movement frames ratio is 18:1 
THRESHOLD = 3
