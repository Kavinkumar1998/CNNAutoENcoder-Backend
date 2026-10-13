import numpy as np
import tensorflow as tf
import keras


def load_and_normalize_data_Set():

    # 1. LOAD AND PREPROCESS MNIST (Grayscale 28x28)
    (mnist_train, _), (mnist_test, _) =  keras.datasets.mnist.load_data()
    # Normalize to [0, 1] range
    mnist_train = mnist_train.astype("float32") / 255.0
    mnist_test = mnist_test.astype("float32") / 255.0
    # Add a trailing channel dimension -> becomes (60000, 28, 28, 1)
    mnist_train = np.expand_dims(mnist_train, axis=-1)
    mnist_test = np.expand_dims(mnist_test, axis=-1)
    # Resize to 32x32 to match CIFAR
    mnist_train_32 = tf.image.resize(mnist_train, [32, 32]).numpy()
    mnist_test_32 = tf.image.resize(mnist_test, [32, 32]).numpy()
    # Convert 1-channel grayscale to 3-channel RGB (replicates gray across R, G, B)
    mnist_train_rgb = tf.image.grayscale_to_rgb(tf.convert_to_tensor(mnist_train_32)).numpy()
    mnist_test_rgb = tf.image.grayscale_to_rgb(tf.convert_to_tensor(mnist_test_32)).numpy()
    # 2. LOAD AND PREPROCESS CIFAR-10 (RGB 32x32)
    (cifar_train, _), (cifar_test, _) = keras.datasets.cifar10.load_data()
    # Normalize to [0, 1] range (already 32x32x3)
    cifar_train = cifar_train.astype("float32") / 255.0
    cifar_test = cifar_test.astype("float32") / 255.0
    # 3. MERGE DATASETS & ADD GAUSSIAN NOISE
    # Concatenate Clean Targets
    X_train_clean = np.concatenate([mnist_train_rgb, cifar_train], axis=0)
    X_test_clean = np.concatenate([mnist_test_rgb, cifar_test], axis=0)
    # Shuffle the combined clean datasets synchronously
    train_indices = np.random.permutation(len(X_train_clean))
    X_train_clean = X_train_clean[train_indices]
    # Generate Noisy Inputs
    X_train_noisy = add_gaussian_noise(X_train_clean, noise_factor=0.2)
    X_test_noisy = add_gaussian_noise(X_test_clean, noise_factor=0.2)
    return X_train_clean, X_test_clean, X_train_noisy, X_test_noisy

## Define a function to add Gaussian Noise
def add_gaussian_noise(images, noise_factor=0.2):
    noise = np.random.normal(loc=0.0, scale=noise_factor, size=images.shape)
    noisy_images = images + noise
    return np.clip(noisy_images, 0.0, 1.0) # Keep pixel values valid between 0 and 1




def run_denoising_pipeline(input_image, model):
    """
    Accepts ANY image size and channel depth, denoises it using 
    the 3-channel 32x32 model, and returns it in its original format.
    """
    # 1. Capture original metadata (Shape and Channels)
    original_height = input_image.shape[0]
    original_width = input_image.shape[1]
    original_channels = input_image.shape[-1]
    
    # 2. Channel Normalization (Handle Grayscale vs Colour)
    if original_channels == 1:
        processed_input = tf.image.grayscale_to_rgb(input_image)
    else:
        processed_input = input_image
        
    # 3. Size Normalization: Force it to 32x32 so the model doesn't crash
    processed_input = tf.image.resize(processed_input, [32, 32])
        
    # Add batch dimension for inference: (1, 32, 32, 3)
    batch_input = tf.expand_dims(processed_input, axis=0)
    
    # 4. Model Prediction
    denoised_batch = model(batch_input)
    denoised_image = tf.squeeze(denoised_batch, axis=0) # Shape is now (32, 32, 3)
    
    # 5. Restore Original Size
    denoised_image = tf.image.resize(denoised_image, [original_height, original_width])
    
    # 6. Restore Original Channel Depth
    if original_channels == 1:
        return tf.image.rgb_to_grayscale(denoised_image)
    else:
        return denoised_image
