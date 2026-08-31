# Raga

AI song discovery with celebrity image recognition, facial emotion detection, browser voice search, text search, and Spotify links.

## Run with the trained models

Use Python 3.12 or 3.13 on Windows. Python 3.14 does not currently have a compatible TensorFlow build for these `.h5` models.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:SPOTIPY_CLIENT_ID = "your-client-id"
$env:SPOTIPY_CLIENT_SECRET = "your-client-secret"
python app.py
```

Open `http://127.0.0.1:5000`.

Keep Spotify credentials in environment variables. Do not commit them to source control.

## Deploy on Render

Create a new **Web Service** from this GitHub repository. Render will use the
included `render.yaml` configuration. Add these environment variables in the
Render dashboard:

- `SPOTIPY_CLIENT_ID`
- `SPOTIPY_CLIENT_SECRET`

The app listens on Render's `PORT` through Gunicorn. The `/health` endpoint is
a liveness check and reports whether Spotify credentials are configured.

## Retrain emotion recognition

Install the Kaggle CLI and authenticate with your own Kaggle API token. The
download command must match the FER-2013 dataset slug available to your Kaggle
account. Export or arrange it as `emotion_data/train/<emotion>/*.jpg` and
`emotion_data/test/<emotion>/*.jpg`, then run:

```powershell
python -m pip install kaggle
kaggle datasets download -d msambare/fer2013 -p emotion_data --unzip
python train_emotion.py --data emotion_data --epochs 30
```

The script creates a validation split, uses augmentation and early stopping,
saves the best model to `face_model.h5`, and prints held-out test accuracy.
