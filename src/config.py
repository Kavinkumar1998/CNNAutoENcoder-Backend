from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / 'model'
MODEL_PATH = MODEL_DIR / 'DenoisingAutoencoder.h5'

# Data configuration
IMG_ROWS = 32
IMG_COLS = 32
IMG_HEIGHT = IMG_ROWS
IMG_WIDTH = IMG_COLS
CHANNELS = 3
INPUT_SHAPE = (IMG_ROWS, IMG_COLS, CHANNELS)
NOISE_FACTOR = 0.2

# Hyperparameters
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001
OPTIMIZER = 'adam'
LOSS = 'mean_squared_error'
METRICS = ['accuracy']