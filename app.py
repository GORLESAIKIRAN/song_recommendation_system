import os
import json
import uuid
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename

ROOT = Path(__file__).resolve().parent
UPLOAD_FOLDER = ROOT / "uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
_emotion_model = None
_hero_model = None
_class_indices = None
_spotify = None
EMOTION_LABELS = {
    "angry": "Angry",
    "disgusted": "Disgust",
    "disgust": "Disgust",
    "fear": "Fearful",
    "fearful": "Fearful",
    "happy": "Happy",
    "neutral": "Neutral",
    "sad": "Sad",
    "surprise": "Surprise",
    "surprised": "Surprise",
}
CANONICAL_EMOTION_LABELS = (
    "Angry", "Disgust", "Fearful", "Happy", "Neutral", "Sad", "Surprise"
)
FERPLUS_LABELS = (
    "Neutral", "Happy", "Surprise", "Sad",
    "Angry", "Disgust", "Fearful", "Neutral"
)
_emotion_onnx = None


def _build_label_lookup(labels):
    return {int(index): name for name, index in labels.items()}


def _load_class_indices():
    global _class_indices
    if _class_indices is None:
        import numpy as np
        _class_indices = np.load(
            ROOT / "class_indices.npy", allow_pickle=True
        ).item()
    return _class_indices


def _extract_face_candidates_from_array(image_array):
    import cv2
    if image_array is None or image_array.size == 0:
        return []

    gray = image_array
    if len(image_array.shape) == 3:
        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)

    cascade_path = (
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        return [image_array]

    candidate_faces = []
    for scale_factor, min_neighbors in ((1.1, 5), (1.08, 3), (1.05, 3)):
        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=(40, 40),
        )
        if len(faces) == 0:
            continue
        for x, y, width, height in faces:
            padding = int(max(width, height) * 0.2)
            x0 = max(0, x - padding)
            y0 = max(0, y - padding)
            x1 = min(gray.shape[1], x + width + padding)
            y1 = min(gray.shape[0], y + height + padding)
            crop = image_array[y0:y1, x0:x1]
            if crop.size:
                candidate_faces.append(crop)
        break

    if not candidate_faces:
        return [image_array]
    return candidate_faces


def _predict_from_array(image_array):
    import cv2
    from PIL import Image
    from tensorflow.keras.preprocessing import image as keras_image
    import numpy as np

    candidates = _extract_face_candidates_from_array(image_array)
    best_label = "Unknown"
    best_confidence = 0.0

    for candidate in candidates:
        if len(candidate.shape) == 3 and candidate.shape[2] == 3:
            rgb_image = cv2.cvtColor(candidate, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = candidate

        pil_image = Image.fromarray(rgb_image.astype("uint8"))
        resized = pil_image.resize((224, 224))
        values = np.expand_dims(
            keras_image.img_to_array(resized) / 255.0, axis=0
        )
        predictions = _load_hero_model().predict(values, verbose=0)[0]
        label_by_index = _build_label_lookup(_load_class_indices())
        index = int(np.argmax(predictions))
        confidence = float(predictions[index])
        label = label_by_index.get(index, "Unknown")

        if confidence > best_confidence:
            best_confidence = confidence
            best_label = label

    return best_label, best_confidence


def _load_hero_model():
    global _hero_model
    if _hero_model is None:
        from tensorflow.keras.models import load_model
        _hero_model = load_model(ROOT / "hero_recognition.h5", compile=False)
    return _hero_model


def _load_emotion_model():
    global _emotion_model
    if _emotion_model is None:
        from tensorflow.keras.models import load_model
        _emotion_model = load_model(ROOT / "face_model.h5", compile=False)
    return _emotion_model


def _load_ferplus_model():
    global _emotion_onnx
    if _emotion_onnx is None:
        import onnxruntime as ort
        _emotion_onnx = ort.InferenceSession(
            str(ROOT / "emotion-ferplus-8.onnx"),
            providers=["CPUExecutionProvider"],
        )
    return _emotion_onnx


def _predict_emotion(gray):
    import cv2
    import numpy as np

    onnx_path = ROOT / "emotion-ferplus-8.onnx"
    if onnx_path.exists():
        session = _load_ferplus_model()
        resized = cv2.resize(gray, (64, 64)).astype("float32") / 255.0
        output = session.run(
            None, {session.get_inputs()[0].name: resized[None, None, :, :]}
        )[0][0]
        probabilities = np.exp(output - np.max(output))
        probabilities /= probabilities.sum()
        index = int(np.argmax(probabilities))
        return FERPLUS_LABELS[index], float(probabilities[index])

    model = _load_emotion_model()
    face = cv2.resize(gray, (48, 48)).astype("float32") / 255.0
    prediction = model.predict(
        np.expand_dims(face, axis=(0, -1)), verbose=0
    )[0]
    index = int(np.argmax(prediction))
    return _emotion_labels_for_model(model)[index], float(prediction[index])


def _emotion_labels_for_model(model):
    labels_path = ROOT / "emotion_labels.json"
    if labels_path.exists():
        labels = [
            EMOTION_LABELS.get(name.lower(), name.title())
            for name in json.loads(labels_path.read_text(encoding="utf-8"))
        ]
    else:
        labels = list(CANONICAL_EMOTION_LABELS)
    output_count = int(model.output_shape[-1])
    if output_count != len(labels):
        raise RuntimeError(
            f"Emotion model has {output_count} outputs; "
            f"expected {len(labels)}."
        )
    return labels


def _get_spotify():
    global _spotify
    if _spotify is None:
        client_id = os.getenv("SPOTIPY_CLIENT_ID")
        client_secret = os.getenv("SPOTIPY_CLIENT_SECRET")
        if not client_id or not client_secret:
            raise RuntimeError(
                "Spotify credentials are not configured. Set "
                "SPOTIPY_CLIENT_ID "
                "and SPOTIPY_CLIENT_SECRET."
            )
        import spotipy
        from spotipy.oauth2 import SpotifyClientCredentials
        auth = SpotifyClientCredentials(
            client_id=client_id, client_secret=client_secret
        )
        _spotify = spotipy.Spotify(auth_manager=auth)
    return _spotify


def _track(item):
    images = item.get("album", {}).get("images", [])
    return {
        "name": item.get("name", "Unknown track"),
        "artist": ", ".join(
            artist.get("name", "Unknown artist")
            for artist in item.get("artists", [])
        ),
        "url": item.get("external_urls", {}).get("spotify", ""),
        "album_art": images[0]["url"] if images else "",
        "preview_url": item.get("preview_url"),
    }


def _spotify_tracks(query, limit=12):
    try:
        results = _get_spotify().search(q=query, type="track", limit=limit)
        return [
            _track(item)
            for item in results.get("tracks", {}).get("items", [])
        ]
    except Exception as spotify_error:
        # Spotify may block Web API calls when the app owner lacks Premium.
        try:
            return _fallback_tracks(query, limit)
        except Exception:
            raise spotify_error


def _fallback_tracks(query, limit=12):
    import json
    from urllib.parse import quote
    from urllib.request import urlopen

    endpoint = "https://itunes.apple.com/search?media=music&limit="
    endpoint += str(limit) + "&term=" + quote(query)
    with urlopen(endpoint, timeout=10) as response:
        data = json.load(response)
    tracks = []
    for item in data.get("results", [])[:limit]:
        artist = item.get("artistName", "Unknown artist")
        name = item.get("trackName", "Unknown track")
        tracks.append({
            "name": name,
            "artist": artist,
            "url": "https://open.spotify.com/search/" + quote(
                f"{name} {artist}"
            ),
            "album_art": item.get("artworkUrl100", "").replace(
                "100x100", "300x300"
            ),
            "preview_url": item.get("previewUrl"),
        })
    if tracks:
        return tracks

    endpoint = "https://api.deezer.com/search?q=" + quote(query)
    with urlopen(endpoint, timeout=10) as response:
        data = json.load(response)
    tracks = []
    for item in data.get("data", [])[:limit]:
        artist = item.get("artist", {}).get("name", "Unknown artist")
        tracks.append({
            "name": item.get("title", "Unknown track"),
            "artist": artist,
            "url": "https://open.spotify.com/search/" + quote(
                f"{item.get('title', '')} {artist}"
            ),
            "album_art": item.get("album", {}).get("cover_medium", ""),
            "preview_url": item.get("preview"),
        })
    if tracks:
        return tracks

    endpoint = "https://itunes.apple.com/search?media=music&limit="
    endpoint += str(limit) + "&term=" + quote(query + " songs")
    with urlopen(endpoint, timeout=10) as response:
        data = json.load(response)
    for item in data.get("results", [])[:limit]:
        artist = item.get("artistName", "Unknown artist")
        name = item.get("trackName", "Unknown track")
        tracks.append({
            "name": name,
            "artist": artist,
            "url": "https://open.spotify.com/search/" + quote(
                f"{name} {artist}"
            ),
            "album_art": item.get("artworkUrl100", "").replace(
                "100x100", "300x300"
            ),
            "preview_url": item.get("previewUrl"),
        })
    return tracks


def _allowed(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() \
        in ALLOWED_EXTENSIONS


def _error(message, status=400):
    return jsonify({"error": message}), status


@app.route("/")
def home():
    return send_from_directory(ROOT, "index.html")


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route("/health")
def health():
    configured = bool(
        os.getenv("SPOTIPY_CLIENT_ID")
        and os.getenv("SPOTIPY_CLIENT_SECRET")
    )
    return jsonify({
        "ok": True,
        "spotify_configured": configured,
        "tensorflow_available": _tensorflow_available(),
        "onnx_emotion_available": (ROOT / "emotion-ferplus-8.onnx").exists(),
    })


def _tensorflow_available():
    try:
        import tensorflow  # noqa: F401
    except ImportError:
        return False
    return True


@app.route("/predict", methods=["POST"])
def predict_celebrity():
    file = request.files.get("file")
    if file is None or not file.filename:
        return _error("Choose an image first.")
    if not _allowed(file.filename):
        return _error("Use a PNG, JPG, JPEG, or WEBP image.")
    filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
    path = UPLOAD_FOLDER / filename
    try:
        file.save(path)
        import cv2

        raw_image = cv2.imread(str(path))
        if raw_image is None:
            raise ValueError("The uploaded image could not be read.")

        hero, confidence = _predict_from_array(raw_image)
        return jsonify({
            "hero": hero,
            "confidence": float(confidence),
            "img_path": f"/uploads/{filename}",
        })
    except Exception as exc:
        return _error(f"Image recognition failed: {exc}", 500)
    finally:
        path.unlink(missing_ok=True)


@app.route("/detect_emotion", methods=["POST"])
def detect_emotion():
    file = request.files.get("image")
    if file is None:
        return _error("No camera image was received.")
    try:
        import cv2
        import numpy as np
        data = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if frame is None:
            return _error("The camera image could not be read.")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        enlarged = cv2.resize(gray, None, fx=1.5, fy=1.5)
        detection_scale = 1.5
        faces = cascade.detectMultiScale(
            enlarged, 1.08, 4, minSize=(45, 45)
        )
        if len(faces) == 0:
            detection_scale = 1
            faces = cascade.detectMultiScale(
                gray, 1.05, 3, minSize=(30, 30)
            )
        if len(faces) == 0:
            x, y, width, height = 0, 0, gray.shape[1], gray.shape[0]
        else:
            x, y, width, height = faces[0]
            x, y, width, height = (
                int(x / detection_scale), int(y / detection_scale),
                int(width / detection_scale), int(height / detection_scale)
            )
            padding = int(max(width, height) * 0.15)
            x = max(0, x - padding)
            y = max(0, y - padding)
            width = min(gray.shape[1] - x, width + padding * 2)
            height = min(gray.shape[0] - y, height + padding * 2)
        face = cv2.resize(gray[y:y + height, x:x + width], (48, 48))
        face = face.astype("float32") / 255.0
        emotion, confidence = _predict_emotion(gray[y:y + height, x:x + width])
        return jsonify({
            "emotion": emotion,
            "confidence": confidence,
        })
    except Exception as exc:
        return _error(f"Emotion detection failed: {exc}", 500)


@app.route("/recommend_songs", methods=["POST"])
def recommend_by_emotion():
    data = request.get_json(silent=True) or {}
    emotion = str(data.get("emotion", "")).strip()
    language = str(data.get("language", "")).strip()
    if not emotion or not language:
        return _error("Emotion and language are required.")
    try:
        language_queries = {
            "Telugu": f"{emotion} Telugu Tollywood film songs",
            "Hindi": f"{emotion} Hindi Bollywood film songs",
            "English": f"{emotion} English songs",
        }
        query = language_queries.get(
            language, f"{emotion} {language} songs"
        )
        tracks = _spotify_tracks(query)
        return jsonify({
            "tracks": tracks,
            "title": f"{emotion} {language} recommendations",
            "language": language,
        })
    except Exception as exc:
        return _error(str(exc), 503)


@app.route("/recommend_song", methods=["POST"])
def recommend_by_keyword():
    data = request.get_json(silent=True) or {}
    keyword = str(data.get("keyword", "")).strip()
    language = str(data.get("language", "")).strip()
    if not keyword:
        return _error("Enter a celebrity, mood, or song keyword.")
    try:
        query = f"{keyword} {language} songs" if language else keyword
        tracks = _spotify_tracks(query)
        if not tracks:
            return _error("No songs found for that search.", 404)
        return jsonify({
            "tracks": tracks,
            "title": f"{keyword} {language} songs" if language
            else f"Songs related to {keyword}",
        })
    except Exception as exc:
        return _error(str(exc), 503)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)
