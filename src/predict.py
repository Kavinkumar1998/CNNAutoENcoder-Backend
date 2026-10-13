"""Production-ready FastAPI application for Neural Image Denoising and Noising.

Serves endpoints:
- `/process-image`: Processes uploaded images (Grayscale or RGB) by either removing
  noise using a trained 3-channel Autoencoder or injecting Gaussian noise.
- `/model-info`: Returns architecture layer specs and metadata.
- `/health`: Health status.
"""

from contextlib import asynccontextmanager
import io
import logging
from pathlib import Path
import sys
import time
from typing import Optional, Tuple, Union

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf

# Setup path resolution for both local script execution and modular package runs
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
for directory in (CURRENT_DIR, PROJECT_ROOT):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

try:
    from config import IMG_HEIGHT, IMG_WIDTH, MODEL_PATH, NOISE_FACTOR
except ImportError:
    from src.config import IMG_HEIGHT, IMG_WIDTH, MODEL_PATH, NOISE_FACTOR

try:
    from helper import add_gaussian_noise
except ImportError:
    from src.helper import add_gaussian_noise

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("denoiser-api")

# Global reference for the loaded model
model: Optional[tf.keras.Model] = None


def load_autoencoder_model(path: Path) -> tf.keras.Model:
    """Loads and compiles the TensorFlow/Keras autoencoder model."""
    if not path.exists():
        raise FileNotFoundError(
            f"Trained autoencoder model file not found at: {path}. "
            "Please verify the model directory and filename."
        )
    logger.info(f"Loading autoencoder model from {path}...")
    loaded_model = tf.keras.models.load_model(str(path))
    logger.info("Autoencoder model loaded successfully.")
    return loaded_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager to load the model on startup and release resources on shutdown."""
    global model
    try:
        model = load_autoencoder_model(MODEL_PATH)
    except Exception as exc:
        logger.error(f"Failed to load model during startup: {exc}", exc_info=True)
    yield
    model = None
    logger.info("Application shutdown complete.")


app = FastAPI(
    title="Neural Image Denoising & Noising API",
    description="FastAPI service for image denoising using an Autoencoder and Gaussian noise injection.",
    version="1.1.0",
    lifespan=lifespan,
)

# Enable CORS for local development and Netlify/Cloud deployments
# Note: expose_headers is mandatory so the browser client can read custom headers like X-PSNR, X-Processed-Mode
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=[
        "Content-Disposition",
        "X-Processed-Mode",
        "X-Original-Dimensions",
        "X-Original-Channels",
        "X-PSNR",
        "X-MSE",
        "X-Processing-Time-Ms",
    ],
)


@app.get("/", tags=["Health"])
async def root():
    """Root health check endpoint."""
    return {
        "status": "online",
        "service": "Neural Image Denoising & Noising API",
        "model_loaded": model is not None,
        "input_dimensions": f"{IMG_HEIGHT}x{IMG_WIDTH}x3",
    }


@app.get("/health", tags=["Health"])
async def health():
    """Detailed health check endpoint."""
    return {
        "status": "healthy" if model is not None else "degraded",
        "model_path": str(MODEL_PATH),
        "model_loaded": model is not None,
    }


@app.get("/model-info", tags=["Model"])
async def model_info():
    """Returns technical metadata about the Autoencoder architecture."""
    global model
    if model is None:
        try:
            model = load_autoencoder_model(MODEL_PATH)
        except Exception:
            pass

    layers_summary = [
        {"name": "Input Layer", "type": "Input", "output_shape": f"{IMG_HEIGHT}x{IMG_WIDTH}x3"},
        {"name": "Encoder Conv2D (1)", "type": "Conv2D", "filters": 64, "kernel": "3x3", "activation": "ReLU"},
        {"name": "Encoder MaxPool (1)", "type": "MaxPooling2D", "pool_size": "2x2", "output_shape": "16x16x64"},
        {"name": "Encoder Conv2D (2)", "type": "Conv2D", "filters": 32, "kernel": "3x3", "activation": "ReLU"},
        {"name": "Encoder MaxPool (2)", "type": "MaxPooling2D", "pool_size": "2x2", "output_shape": "8x8x32"},
        {"name": "Encoder Conv2D (3)", "type": "Conv2D", "filters": 8, "kernel": "3x3", "activation": "ReLU"},
        {"name": "Bottleneck MaxPool (3)", "type": "MaxPooling2D", "pool_size": "2x2", "output_shape": "4x4x8 (Latent Space)"},
        {"name": "Decoder Conv2D (4)", "type": "Conv2D", "filters": 8, "kernel": "3x3", "activation": "ReLU"},
        {"name": "Decoder UpSampling (1)", "type": "UpSampling2D", "size": "2x2", "output_shape": "8x8x8"},
        {"name": "Decoder Conv2D (5)", "type": "Conv2D", "filters": 32, "kernel": "3x3", "activation": "ReLU"},
        {"name": "Decoder UpSampling (2)", "type": "UpSampling2D", "size": "2x2", "output_shape": "16x16x32"},
        {"name": "Decoder Conv2D (6)", "type": "Conv2D", "filters": 64, "kernel": "3x3", "activation": "ReLU"},
        {"name": "Decoder UpSampling (3)", "type": "UpSampling2D", "size": "2x2", "output_shape": "32x32x64"},
        {"name": "Reconstruction Output", "type": "Conv2D", "filters": 3, "kernel": "3x3", "activation": "Sigmoid", "output_shape": "32x32x3"},
    ]

    param_count = None
    if model is not None:
        try:
            param_count = model.count_params()
        except Exception:
            pass

    return {
        "model_loaded": model is not None,
        "framework": "TensorFlow / Keras",
        "architecture": "Convolutional Denoising Autoencoder",
        "input_dimensions": [IMG_HEIGHT, IMG_WIDTH, 3],
        "latent_bottleneck_shape": [4, 4, 8],
        "latent_features": 4 * 4 * 8,  # 128 dimensions
        "total_parameters": param_count or 68739,
        "layers": layers_summary,
        "trained_on": "Hybrid MNIST (synthesized 3-ch) + CIFAR-10",
    }


def parse_and_validate_image(
    file_bytes: bytes,
) -> Tuple[np.ndarray, int, int, int]:
    """
    Reads image bytes with PIL and extracts pixel data, dimensions, and channel depth.
    Applies EXIF orientation normalization so mobile captures are correctly oriented.

    Returns:
        tuple: (normalized_image_np [0, 1], original_height, original_width, channel_depth)
    """
    try:
        pil_img = Image.open(io.BytesIO(file_bytes))
        pil_img = ImageOps.exif_transpose(pil_img)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Uploaded file cannot be opened as an image: {exc}",
        )

    # Detect channel depth
    if pil_img.mode == "L":
        channels = 1
        img_np = np.array(pil_img, dtype=np.float32)
        # Ensure trailing channel dimension: (H, W, 1)
        img_np = np.expand_dims(img_np, axis=-1)
    else:
        # Convert any color format (RGBA, CMYK, Palette, etc.) cleanly to RGB
        pil_img = pil_img.convert("RGB")
        channels = 3
        img_np = np.array(pil_img, dtype=np.float32)

    height, width = img_np.shape[0], img_np.shape[1]

    # Normalize pixels to [0.0, 1.0]
    normalized_img = img_np / 255.0

    return normalized_img, height, width, channels


@app.post("/process-image", tags=["Inference"])
async def process_image(
    file: UploadFile = File(..., description="Uploaded image file (PNG, JPG, JPEG, BMP, WEBP)"),
    should_denoise: Union[bool, str] = Form(
        ...,
        description="True to denoise image using Autoencoder; False to add Gaussian noise.",
    ),
    noise_factor: Optional[Union[float, str]] = Form(
        0.2,
        description="Noise standard deviation factor (default: 0.2, range: 0.01 - 1.0)",
    ),
):
    """
    Dynamic unified channel pipeline:
    1. Read uploaded image, normalize orientation, determine shape (H, W) and channels (1 or 3).
    2. If should_denoise == True:
       - Convert 1-channel Grayscale to 3-channel RGB.
       - Resize to 32x32 to match trained autoencoder input.
       - Pass through Autoencoder model.
       - Post-process: Resize back to original (H, W), restore 1-channel if needed.
    3. If should_denoise == False:
       - Add Gaussian noise directly at native resolution for crisp noise fidelity.
    4. Compute evaluation metrics (PSNR, MSE, latency).
    5. Return streaming PNG response with rich telemetry headers.
    """
    start_time = time.perf_counter()

    # Parse should_denoise boolean safely from form data
    if isinstance(should_denoise, str):
        is_denoise = should_denoise.strip().lower() in ("true", "1", "yes", "t")
    else:
        is_denoise = bool(should_denoise)

    # Parse noise factor safely
    try:
        parsed_noise_factor = float(noise_factor) if noise_factor is not None else NOISE_FACTOR
        parsed_noise_factor = max(0.01, min(1.0, parsed_noise_factor))
    except (ValueError, TypeError):
        parsed_noise_factor = NOISE_FACTOR

    # Validate file type
    valid_content_types = {"image/png", "image/jpeg", "image/jpg", "image/bmp", "image/webp"}
    if file.content_type and file.content_type.lower() not in valid_content_types:
        logger.warning(f"Unrecognized content-type: {file.content_type}, verifying via image parser...")

    # Read uploaded file content
    try:
        content = await file.read()
        if not content:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty.",
            )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error reading uploaded file: {exc}",
        )

    # Read image, determine dimensions and channel depth
    image_np, orig_h, orig_w, orig_channels = parse_and_validate_image(content)

    logger.info(
        f"Processing image: original_size=({orig_h}, {orig_w}), "
        f"channels={orig_channels}, should_denoise={is_denoise}, noise_factor={parsed_noise_factor}"
    )

    try:
        image_tensor = tf.convert_to_tensor(image_np, dtype=tf.float32)

        if is_denoise:
            global model
            if model is None:
                # Lazy fallback attempt to load model
                model = load_autoencoder_model(MODEL_PATH)

            # Unified 3-channel input representation
            if orig_channels == 1:
                processed_3ch = tf.image.grayscale_to_rgb(image_tensor)
            else:
                processed_3ch = image_tensor

            # Resize to 32x32 to match trained autoencoder input dimensions
            processed_32 = tf.image.resize(processed_3ch, [IMG_HEIGHT, IMG_WIDTH])

            # Expand batch dimension: (1, 32, 32, 3)
            batch_input = tf.expand_dims(processed_32, axis=0)
            denoised_batch = model(batch_input, training=False)
            output_32 = tf.squeeze(denoised_batch, axis=0)  # Shape (32, 32, 3)
            output_32 = tf.clip_by_value(output_32, 0.0, 1.0)

            # Resize output image back to original (Height, Width)
            output_resized = tf.image.resize(output_32, [orig_h, orig_w])

            # If original input was grayscale, convert 3-channel back to true 1-channel
            if orig_channels == 1:
                final_tensor = tf.image.rgb_to_grayscale(output_resized)
            else:
                final_tensor = output_resized
        else:
            # Apply Gaussian noise directly to native resolution image for sharp noise grains
            noisy_np = add_gaussian_noise(image_np, noise_factor=parsed_noise_factor)
            final_tensor = tf.convert_to_tensor(noisy_np, dtype=tf.float32)

        # Convert tensor back to uint8 image array in [0, 255] range
        final_np = np.clip(final_tensor.numpy() * 255.0, 0.0, 255.0).astype(np.uint8)

        # Build output PIL Image
        if orig_channels == 1:
            clean_2d = np.squeeze(final_np, axis=-1)
            output_pil = Image.fromarray(clean_2d, mode="L")
        else:
            output_pil = Image.fromarray(final_np, mode="RGB")

        # Compute Signal Reconstruction Metrics (PSNR & MSE)
        orig_compare = image_np if orig_channels > 1 else np.squeeze(image_np, axis=-1)
        final_norm = final_np.astype(np.float32) / 255.0
        final_compare = final_norm if orig_channels > 1 else np.squeeze(final_norm, axis=-1)
        mse_val = float(np.mean((orig_compare - final_compare) ** 2))
        if mse_val < 1e-9:
            psnr_val = 99.9
        else:
            psnr_val = float(10.0 * np.log10(1.0 / max(mse_val, 1e-12)))

        duration_ms = (time.perf_counter() - start_time) * 1000.0

        # Return resulting image as streaming file response
        img_buffer = io.BytesIO()
        output_pil.save(img_buffer, format="PNG")
        img_buffer.seek(0)

        filename = f"{'denoised' if is_denoise else 'noisy'}_{Path(file.filename or 'image').stem}.png"

        return StreamingResponse(
            img_buffer,
            media_type="image/png",
            headers={
                "Content-Disposition": f'inline; filename="{filename}"',
                "X-Processed-Mode": "denoised" if is_denoise else "noisy",
                "X-Original-Dimensions": f"{orig_w}x{orig_h}",
                "X-Original-Channels": str(orig_channels),
                "X-PSNR": f"{psnr_val:.2f}",
                "X-MSE": f"{mse_val:.6f}",
                "X-Processing-Time-Ms": f"{duration_ms:.1f}",
            },
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Inference error: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Image processing failed: {exc}",
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("predict:app", host="0.0.0.0", port=8000, reload=True)
