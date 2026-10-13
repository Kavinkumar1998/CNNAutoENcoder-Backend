# NeuroClean Studio • Neural Image Denoising & Synthesis Web Application

A production-grade deep learning application powered by **TensorFlow / Keras**, a high-performance **FastAPI** backend, and a modern **React 19 + Tailwind CSS** studio interface.

---

## 🌟 Key Capabilities & Architecture

- **Convolutional Autoencoder Model**: Pretrained 3-channel ($32 \times 32 \times 3$) model capable of removing noise from both Grayscale (e.g. MNIST) and RGB (e.g. CIFAR-10) images.
- **Dynamic Unified Channel Pipeline**:
  - Dynamically detects Grayscale (1-channel) vs. RGB (3-channel) inputs.
  - Automatically converts 1-channel to 3-channel for inference (`tf.image.grayscale_to_rgb`).
  - Normalizes EXIF orientation tags so smartphone photos are right-side-up.
  - Resizes to $32 \times 32$ for model input.
  - Inverses operations post-processing: resizes back to the original resolution and restores the original channel depth.
- **Dual Pipeline Operations**:
  - **Denoise Image**: Passes images through the convolutional autoencoder to filter out noise.
  - **Synthesize Noise (Gaussian)**: Injects Gaussian normal noise ($\mathcal{N}(0, \sigma^2)$) with interactive variance control ($\sigma \in [0.05, 0.50]$) at native resolution for realistic grain.
- **Quantitative Telemetry Metrics**:
  - Computes **PSNR (Peak Signal-to-Noise Ratio)** and **MSE (Mean Squared Error)** on every request.
  - Returns end-to-end and server model latency.
- **Interactive Visual Comparison Suite**:
  - **Split Slider**: Pixel-perfect before/after comparison using CSS `clip-path` (no stretching or misalignment).
  - **Side-by-Side Inspection**: Synchronized high-resolution inspection.
  - **Noise Residual Delta Map**: Real-time visual subtraction ($|I_{in} - I_{out}| \times 4$) isolating noise components.
  - **Cycle Pipeline ("Use as Input")**: Seamlessly feed synthesized noisy images directly back into the denoiser.
  - **Clipboard Support**: Paste screenshots directly with `Ctrl+V`.
  - **Curated Demo Presets**: Instant one-click samples (MNIST "8", CIFAR Car, Clean Gradient, Cellular Texture).

---

## 📁 Project Directory Structure

```
Denoisingwebapp/
├── backend/
│   ├── model/
│   │   └── DenoisingAutoencoder.h5       # Trained TensorFlow/Keras model
│   ├── src/
│   │   ├── config.py                     # Configuration constants (paths, shapes, noise factor)
│   │   ├── helper.py                     # Gaussian noise injection & inference utilities
│   │   ├── predict.py                    # Production-ready FastAPI service (/process-image, /model-info)
│   │   ├── train.py                      # Model training script
│   │   └── features/
│   │       └── helperFunctions.py        # Dataset loading & helpers
│   ├── tests/
│   │   └── test_api.py                   # Automated API integration tests
│   ├── app.py                            # Root FastAPI entrypoint
│   ├── requirements.txt                  # Python dependencies (UTF-8, fastapi, multipart, etc.)
│   └── README.md
├── frontend/
│   ├── src/
│   │   ├── App.jsx                       # Main React studio component
│   │   ├── main.jsx                      # React application entry point
│   │   └── index.css                     # Tailwind CSS styles & custom utilities
│   ├── package.json                      # Frontend dependencies (React 19, Lucide, Tailwind 4)
│   ├── vite.config.js                    # Vite build and proxy configuration
│   └── index.html                        # HTML shell
```

---

## 🚀 Getting Started

### 1. Backend Setup (FastAPI)

1. Activate your Python virtual environment:
   ```powershell
   cd backend
   .\.venv\Scripts\Activate.ps1
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Start the FastAPI server:
   ```bash
   python app.py
   # Or using uvicorn directly:
   uvicorn app:app --reload --host 0.0.0.0 --port 8000
   ```
   Interactive Swagger docs are live at [http://localhost:8000/docs](http://localhost:8000/docs).

4. Run the automated test suite:
   ```bash
   python tests/test_api.py
   ```

---

### 2. Frontend Setup (React + Vite)

1. Navigate to the `frontend/` directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Launch development server:
   ```bash
   npm run dev
   ```
   Open your browser at [http://localhost:5173](http://localhost:5173).

4. Build for production:
   ```bash
   npm run build
   ```

---

## 🔌 API Endpoints Reference

### `POST /process-image`
Accepts multipart form-data:
- `file` (*UploadFile*): Image file (PNG, JPG, JPEG, BMP, WEBP).
- `should_denoise` (*bool*): `true` to denoise via Autoencoder, `false` to inject Gaussian noise.
- `noise_factor` (*float*, optional): Standard deviation factor for noise (default `0.2`, range `0.01` to `1.0`).

**Response Headers**:
- `X-Processed-Mode`: `denoised` or `noisy`
- `X-Original-Dimensions`: `WxH`
- `X-Original-Channels`: `1` or `3`
- `X-PSNR`: Calculated Peak Signal-to-Noise Ratio (dB)
- `X-MSE`: Calculated Mean Squared Error
- `X-Processing-Time-Ms`: Inference duration in milliseconds

### `GET /model-info`
Returns architectural metadata (layers, filters, latent bottleneck dimensions, parameter count).

### `GET /health`
Returns health check and model loading status.
