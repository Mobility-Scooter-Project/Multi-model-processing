# Current implementation deflates the number of pose to match the number of move points
# Another implementation may be to inflate the number of move sequences to match number of pose sequences
BATCH_SIZE = 32
RANDOM_SEED = 42
POSE_N_FEATURES = 18
MOVE_N_FEATURES = 6
TEST_SIZE = 0.15
LEARNING_RATE = 1e-4
DEFLATION_FACTOR = 18 # Video frames to movement frames ratio is 18:1 
THRESHOLD = 2
