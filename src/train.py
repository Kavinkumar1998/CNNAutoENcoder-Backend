from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Sequential

from src import config
from src.features.helperFunctions import load_and_normalize_data_Set


def train_model():
    # Define the model architecture
    model = Sequential([
        # Encoder
        Conv2D( filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', input_shape=config.INPUT_SHAPE,  ),
        MaxPooling2D(pool_size=(2, 2), padding='same'),
        Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(2, 2), padding='same'),
        Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(2, 2), padding='same'),
        # Decoder
        Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'),
        UpSampling2D(size=(2, 2)),
        Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'),
        UpSampling2D(size=(2, 2)),
        Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'),
        UpSampling2D(size=(2, 2)),
        Conv2D(filters=3, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'),
    ])

    model.compile(optimizer=config.OPTIMIZER, loss=config.LOSS, metrics=config.METRICS)
    return model


def run_model_pipeline():
    # Load and preprocess the data
    X_train_clean, X_test_clean, X_train_noisy, X_test_noisy = load_and_normalize_data_Set()

    print('Compiling model...')
    model = train_model()

    print('Training model...')
    model.fit(
        X_train_noisy,
        X_train_clean,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        validation_data=(X_test_noisy, X_test_clean),
    )

    model.summary()

    os.makedirs(config.MODEL_DIR, exist_ok=True)
    print(f'Saving model to: {config.MODEL_PATH}')
    model.save(config.MODEL_PATH)
    print('Pipeline sequence completed successfully.')


if __name__ == '__main__':
    run_model_pipeline()

