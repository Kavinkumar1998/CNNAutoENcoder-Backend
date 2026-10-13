"""Unit tests for the FastAPI Denoising and Noising API."""

import io
from pathlib import Path
import sys
from PIL import Image
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from starlette.testclient import TestClient

from src.predict import app


def create_dummy_image(width=64, height=48, mode="RGB"):
    """Creates an in-memory PNG image."""
    if mode == "L":
        arr = (np.random.rand(height, width) * 255).astype(np.uint8)
    else:
        arr = (np.random.rand(height, width, 3) * 255).astype(np.uint8)
    img = Image.fromarray(arr, mode=mode)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()


def test_health_check():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "online"
        assert data["model_loaded"] is True


def test_model_info():
    with TestClient(app) as client:
        response = client.get("/model-info")
        assert response.status_code == 200
        data = response.json()
        assert data["framework"] == "TensorFlow / Keras"
        assert "layers" in data
        assert len(data["layers"]) > 5


def test_denoise_rgb_image():
    with TestClient(app) as client:
        img_bytes = create_dummy_image(width=60, height=40, mode="RGB")
        response = client.post(
            "/process-image",
            files={"file": ("test_rgb.png", img_bytes, "image/png")},
            data={"should_denoise": "true"},
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"
        assert "x-psnr" in response.headers
        assert "x-mse" in response.headers
        out_img = Image.open(io.BytesIO(response.content))
        assert out_img.size == (60, 40)
        assert out_img.mode == "RGB"


def test_noise_rgb_image():
    with TestClient(app) as client:
        img_bytes = create_dummy_image(width=50, height=50, mode="RGB")
        response = client.post(
            "/process-image",
            files={"file": ("test_noise.png", img_bytes, "image/png")},
            data={"should_denoise": "false", "noise_factor": "0.3"},
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"
        assert "x-psnr" in response.headers
        out_img = Image.open(io.BytesIO(response.content))
        assert out_img.size == (50, 50)
        assert out_img.mode == "RGB"


def test_denoise_grayscale_image():
    with TestClient(app) as client:
        img_bytes = create_dummy_image(width=28, height=28, mode="L")
        response = client.post(
            "/process-image",
            files={"file": ("test_gray.png", img_bytes, "image/png")},
            data={"should_denoise": "true"},
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"
        out_img = Image.open(io.BytesIO(response.content))
        assert out_img.size == (28, 28)
        assert out_img.mode == "L"


def test_noise_grayscale_image():
    with TestClient(app) as client:
        img_bytes = create_dummy_image(width=32, height=32, mode="L")
        response = client.post(
            "/process-image",
            files={"file": ("test_gray_noise.png", img_bytes, "image/png")},
            data={"should_denoise": "false"},
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"
        out_img = Image.open(io.BytesIO(response.content))
        assert out_img.size == (32, 32)
        assert out_img.mode == "L"


if __name__ == "__main__":
    print("Running tests...")
    test_health_check()
    print("Health check: PASS")
    test_model_info()
    print("Model info: PASS")
    test_denoise_rgb_image()
    print("Denoise RGB: PASS")
    test_noise_rgb_image()
    print("Noise RGB: PASS")
    test_denoise_grayscale_image()
    print("Denoise Grayscale: PASS")
    test_noise_grayscale_image()
    print("Noise Grayscale: PASS")
    print("ALL TESTS PASSED!")
