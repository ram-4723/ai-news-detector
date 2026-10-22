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


try:
    model = load_artifact("model.pkl")
    vectorizer = load_artifact("vectorizer.pkl")
except Exception:
    # NewsAPI is the required primary check. The old classifier is only shown
    # as a secondary estimate, so a missing/corrupt model must not disable it.
    app.logger.exception("Optional classifier artifacts could not be loaded")
    model = vectorizer = None


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


class NewsAPIError(Exception):
    """Raised when a mandatory NewsAPI search cannot be completed."""


def search_newsapi(query):
    api_key = os.getenv("NEWSAPI_KEY", "").strip()
    if not api_key:
        raise NewsAPIError("NewsAPI is required but NEWSAPI_KEY is not configured on the server.")

    try:
        response = requests.get(
            NEWS_API_URL,
            params={"q": query[:500], "pageSize": 5, "sortBy": "relevancy"},
            headers={"X-Api-Key": api_key},
            timeout=15,
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("status") != "ok":
            code = payload.get("code", "unknown")
            message = payload.get("message", "NewsAPI returned an error.")
            raise NewsAPIError(f"NewsAPI error ({code}): {message}")

        articles = [
            {
                "title": article.get("title") or "Untitled article",
                "description": article.get("description") or "",
                "source": (article.get("source") or {}).get("name") or "Unknown source",
                "published_at": article.get("publishedAt") or "",
                "url": article.get("url", ""),
            }
            for article in payload.get("articles", [])[:5]
            if article.get("url", "").startswith(("https://", "http://"))
        ]
        return int(payload.get("totalResults", len(articles))), articles
    except NewsAPIError:
        raise
    except (requests.RequestException, ValueError, TypeError) as exc:
        app.logger.warning("Required NewsAPI search failed: %s", exc)
        raise NewsAPIError("NewsAPI could not be reached. Check the key, plan, and service logs.") from exc


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
        total_results, articles = search_newsapi(full_text)
    except NewsAPIError as exc:
        app.logger.error("NewsAPI is required for analysis: %s", exc)
        status_code = 503 if "not configured" in str(exc) else 502
        return jsonify({"error": str(exc)}), status_code

    classifier_estimate = None
    classifier_probability = None
    if model is not None and vectorizer is not None:
        try:
            transformed = vectorizer.transform([full_text])
            prediction = int(model.predict(transformed)[0])
            probabilities = model.predict_proba(transformed)[0]
            classifier_estimate = "Likely Real" if prediction == 1 else "Likely Fake"
            classifier_probability = round(float(max(probabilities)) * 100, 2)
        except Exception:
            app.logger.exception("Secondary classifier estimate failed")

    # NewsAPI returns article-search results, not a fact-check verdict. Stay
    # inconclusive and show sources for manual comparison instead of claiming
    # that a matching headline proves or disproves the submitted claim.
    if total_results:
        evidence_summary = (
            f"NewsAPI found {total_results} related article(s). This is not enough by itself "
            "to confirm or refute the claim; review the sources below."
        )
    else:
        evidence_summary = "NewsAPI found no matching articles. This does not prove the claim is false."

    return jsonify(
        {
            "fact_check_status": "inconclusive",
            "evidence_summary": evidence_summary,
            "total_results": total_results,
            "articles": articles,
            "classifier_estimate": classifier_estimate,
            "classifier_probability": classifier_probability,
            "ocr_text": extracted_text,
        }
    )


@app.errorhandler(413)
def request_too_large(_error):
    return jsonify({"error": "The request is too large. Images must be smaller than 5 MB."}), 413


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")))
