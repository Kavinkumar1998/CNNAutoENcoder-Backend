"""Helper functions and preprocessing utilities for the Denoising Autoencoder."""

from typing import Tuple, Union
import numpy as np
import tensorflow as tf


def add_gaussian_noise(images: Union[np.ndarray, tf.Tensor], noise_factor: float = 0.2) -> np.ndarray:
    """
    Adds Gaussian (normal) noise to an image or batch of images.
    Values are clipped to the valid [0.0, 1.0] range.
    """
    if isinstance(images, tf.Tensor):
        images_np = images.numpy()
    else:
        images_np = np.asarray(images, dtype=np.float32)

    noise = np.random.normal(loc=0.0, scale=noise_factor, size=images_np.shape)
    noisy_images = images_np + noise
    return np.clip(noisy_images, 0.0, 1.0).astype(np.float32)


def run_denoising_pipeline(input_image: tf.Tensor, model: tf.keras.Model) -> tf.Tensor:
    """
    Accepts ANY image size and channel depth, denoises it using
    the 3-channel 32x32 model, and returns it in its original format.
    """
    original_height = input_image.shape[0]
    original_width = input_image.shape[1]
    original_channels = input_image.shape[-1]

    # Channel Normalization (Handle Grayscale vs Colour)
    if original_channels == 1:
        processed_input = tf.image.grayscale_to_rgb(input_image)
    else:
        processed_input = input_image

    # Size Normalization: Force it to 32x32 to match the model's expected shape
    processed_input = tf.image.resize(processed_input, [32, 32])

    # Add batch dimension for inference: (1, 32, 32, 3)
    batch_input = tf.expand_dims(processed_input, axis=0)

    # Model Prediction
    denoised_batch = model(batch_input, training=False)
    denoised_image = tf.squeeze(denoised_batch, axis=0)  # Shape (32, 32, 3)

    # Restore Original Size
    denoised_image = tf.image.resize(denoised_image, [original_height, original_width])

    # Restore Original Channel Depth
    if original_channels == 1:
        return tf.image.rgb_to_grayscale(denoised_image)
    return denoised_image
