"""Flask application for the AI news detector."""

import base64
import binascii
import io
import logging
import os
import pickle
from pathlib import Path

import pytesseract
import requests
from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError


BASE_DIR = Path(__file__).resolve().parent
MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_TEXT_LENGTH = 20_000
NEWS_API_URL = "https://newsapi.org/v2/everything"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024
app.logger.setLevel(logging.INFO)

# On Windows, allow a local Tesseract installation to be configured without
# hard-coding a machine-specific path in the source.
if os.name == "nt" and os.getenv("TESSERACT_CMD"):
    pytesseract.pytesseract.tesseract_cmd = os.environ["TESSERACT_CMD"]


def load_artifact(filename):
    path = BASE_DIR / filename
    if not path.is_file():
        raise RuntimeError(f"Required model artifact is missing: {filename}")
    # These pickle files are executable Python serialization. Only load model
    # artifacts that are part of this trusted application repository.
    with path.open("rb") as artifact:
        return pickle.load(artifact)


model = load_artifact("model.pkl")
vectorizer = load_artifact("vectorizer.pkl")


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


def extract_image_text(image_value):
    if not image_value:
        return ""
    if not isinstance(image_value, str):
        raise ValueError("The image must be sent as a base64 data URL.")

    encoded = image_value.split(",", 1)[-1]
    try:
        image_bytes = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("The uploaded image data is invalid.") from exc
    if not image_bytes or len(image_bytes) > MAX_IMAGE_BYTES:
        raise ValueError("Upload an image smaller than 5 MB.")

    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            image.verify()
        with Image.open(io.BytesIO(image_bytes)) as image:
            if image.width * image.height > 25_000_000:
                raise ValueError("The image is too large to process.")
            return pytesseract.image_to_string(image.convert("RGB")).strip()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("The uploaded file is not a supported image.") from exc
    except pytesseract.TesseractNotFoundError as exc:
        raise RuntimeError("OCR is unavailable because Tesseract is not installed.") from exc


def find_articles(query):
    api_key = os.getenv("NEWSAPI_KEY", "").strip()
    if not api_key:
        return []

    try:
        response = requests.get(
            NEWS_API_URL,
            params={"q": query[:200], "pageSize": 5, "sortBy": "relevancy"},
            headers={"X-Api-Key": api_key},
            timeout=8,
        )
        response.raise_for_status()
        payload = response.json()
        return [
            {"title": article.get("title") or "Untitled article", "url": article.get("url", "")}
            for article in payload.get("articles", [])[:5]
            if article.get("url", "").startswith(("https://", "http://"))
        ]
    except (requests.RequestException, ValueError) as exc:
        app.logger.warning("NewsAPI lookup failed: %s", exc)
        return []


@app.post("/analyze")
def analyze():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Send a JSON object containing news and/or image."}), 400

    news = data.get("news", "")
    if not isinstance(news, str):
        return jsonify({"error": "News content must be text."}), 400
    news = news.strip()
    if len(news) > MAX_TEXT_LENGTH:
        return jsonify({"error": f"News content must be {MAX_TEXT_LENGTH:,} characters or fewer."}), 413

    try:
        extracted_text = extract_image_text(data.get("image", ""))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except RuntimeError as exc:
        app.logger.error("OCR is not available: %s", exc)
        return jsonify({"error": str(exc)}), 503

    full_text = " ".join(part for part in (news, extracted_text) if part).strip()
    if len(full_text.split()) < 5:
        return jsonify({"error": "Enter at least five words, or upload an image with readable text."}), 422

    try:
        transformed = vectorizer.transform([full_text])
        prediction = int(model.predict(transformed)[0])
        probabilities = model.predict_proba(transformed)[0]
        confidence = round(float(max(probabilities)) * 100, 2)
    except Exception:
        app.logger.exception("Model prediction failed")
        return jsonify({"error": "The analysis model could not process this content."}), 500

    result = "Likely Real" if prediction == 1 else "Likely Fake"
    return jsonify(
        {
            "prediction": result,
            "confidence": confidence,
            "articles": find_articles(full_text),
            "ocr_text": extracted_text,
        }
    )


@app.errorhandler(413)
def request_too_large(_error):
    return jsonify({"error": "The request is too large. Images must be smaller than 5 MB."}), 413


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")))
