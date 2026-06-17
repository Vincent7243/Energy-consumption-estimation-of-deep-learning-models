# EdgeBench Predictor

A web application for predicting deep learning model performance on two hardware platforms: **RTX 3080** (server GPU) and **Jetson Nano** (edge device). Given a model name or an uploaded weight file, it returns predicted **Accuracy**, **Latency**, **Throughput**, and **Energy consumption** using a CatBoost ensemble trained on real benchmark data.

## Features

- Predict performance metrics for 300+ models by name (ResNet, ViT, EfficientNet, Swin, ConvNeXt, and more)
- Support for arbitrary [timm](https://github.com/huggingface/pytorch-image-models) models not in the dataset — features are computed automatically
- Upload custom model files (`.pth`, `.pt`, `.safetensors`, `.bin`, `.ckpt`) for prediction
- Switch between **RTX 3080** and **Jetson Nano** platforms
- Live SSH terminal to Jetson Nano for running real benchmarks
- User authentication with email OTP verification

## Tech Stack

- **Frontend:** Vue 3 single-page app (`index.html`)
- **Backend:** FastAPI (REST API + WebSocket SSH)
- **ML:** CatBoost ensemble, timm + torchinfo for feature extraction
- **Auth:** Email + OTP verification

## Project Structure

```
app/
├── index.html              # Frontend — Vue 3 single-page app
└── backend/
    ├── main.py             # FastAPI backend (REST API + WebSocket SSH)
    ├── train_models.py     # CatBoost training script (52-feature pipeline)
    ├── requirements.txt    # Python dependencies
    ├── .env                # Email credentials (not committed — see setup)
    ├── JetsonNano_model.csv  # Benchmark dataset — Jetson Nano
    └── RTX_3080_results.csv  # Benchmark dataset — RTX 3080
flops_csv2.py               # Script to collect benchmark data on Jetson Nano
```

## Setup

### 1. Install dependencies

```bash
cd app/backend
pip install -r requirements.txt
```

### 2. Train the CatBoost models

Run once to generate the model files used by the API:

```bash
python train_models.py --csv RTX_3080_results.csv --platform rtx
python train_models.py --csv JetsonNano_model.csv --platform jetson
```

This creates `models/rtx/` and `models/jetson/` containing `.cbm` ensemble files and a `meta.json` lookup table.

### 3. Configure email (optional)

Create `app/backend/.env` for OTP email delivery:

```
MAIL_USER=your-email@gmail.com
MAIL_PASS=your-gmail-app-password
```

If these credentials are not configured, OTP codes are printed to the console (development mode).

### 4. Start the backend

```bash
cd app/backend
uvicorn main:app --reload --port 8000
```

### 5. Open the frontend

Open `app/index.html` directly in a browser, or serve it with:

```bash
python -m http.server 5500 --directory app
```

Then open `http://localhost:5500`.

## Usage

### Predict by model name

1. Register or log in (email + OTP verification)
2. Select a platform: **RTX 3080** or **Jetson Nano**
3. Type a model name (e.g. `resnet50`, `vit_base_patch16_224`, `efficientnet_b0`)
4. Click **Predict** — results show Accuracy, Latency (ms), Throughput (img/s), and Energy (J)

Models in the training dataset return results instantly from the lookup table. Models not in the dataset are handled automatically using [timm](https://github.com/huggingface/pytorch-image-models) + torchinfo to compute the required features.

### Predict from a model file

Uploading a `.pth`, `.pt`, `.safetensors`, `.bin`, or `.ckpt` file triggers the backend to:

1. Match the filename against the lookup table
2. If unmatched, load the file and identify the architecture by parameter count (within 2% tolerance)
3. Fall back to feature estimation from the weight structure

### Jetson Nano SSH terminal

Providing SSH credentials (host, username, password) and a model name runs `flops_csv2.py` directly on the board. Output streams in real time via WebSocket. The benchmark can be paused, resumed, or stopped at any time.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/predict` | Predict metrics for a model by name |
| `POST` | `/predict-file` | Predict metrics from an uploaded model file |
| `POST` | `/validate-model` | Check if a model name exists; returns suggestions |
| `GET` | `/models/{platform}` | List all models in the lookup table |
| `POST` | `/auth/register` | Register (sends OTP email) |
| `POST` | `/auth/verify-otp` | Verify OTP and create account |
| `POST` | `/auth/login` | Login with email and password |
| `WS` | `/ws/ssh` | WebSocket — stream Jetson Nano benchmark output |
| `GET` | `/health` | Backend health check |

## ML Pipeline

The predictor uses a **3-seed CatBoost ensemble** with 52 engineered features per model:

- Static architecture features: parameter count, MACs, activations, input size
- Name-derived features: family, size variant (tiny/small/base/large), patch size, depth hint
- Keyword flags for known architecture patterns (ViT, Swin, ConvNeXt, etc.)
- Log-transformed ratios: MACs/param, activations/param, MACs × input area
- Batch size features: optimal BS, max BS (log-transformed)

Latency, Throughput, and Energy targets are log-transformed during training and inverse-transformed at inference.