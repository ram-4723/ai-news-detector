from flask import Flask, render_template, request, jsonify
import pickle
import requests
import pytesseract
from PIL import Image
import base64
import io

app = Flask(__name__)

API_KEY = "2d032abfb1e249bca4299081e978a769"

# Tesseract OCR Path
import os

if os.name == "nt":
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load ML model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))


# Home Page
@app.route("/")
def home():
    return render_template("index.html")


# AI Analysis API
@app.route("/analyze", methods=["POST"])
def analyze():

    data = request.get_json()

    news = data.get("news", "")
    image_data = data.get("image", "")

    extracted_text = ""

    # OCR IMAGE EXTRACTION
    if image_data:

        try:

            image_data = image_data.split(",")[1]

            image_bytes = base64.b64decode(image_data)

            image = Image.open(io.BytesIO(image_bytes))

            extracted_text = pytesseract.image_to_string(image)

        except:
            extracted_text = ""

    # COMBINE TEXT + OCR TEXT
    full_text = (news + " " + extracted_text).strip()

    # VALIDATION
    if len(full_text.split()) < 5:

        return jsonify({
            "prediction": "Please enter proper news content.",
            "confidence": 0,
            "articles": [],
            "ocr_text": extracted_text
        })

    # NLP PROCESSING
    transformed_news = vectorizer.transform([full_text])

    # ML PREDICTION
    prediction = model.predict(transformed_news)[0]

    # CONFIDENCE
    probabilities = model.predict_proba(transformed_news)

    confidence = round(max(probabilities[0]) * 100, 2)

    # FINAL RESULT
    result = "Likely Real" if prediction == 1 else "Likely Fake"

    # NEWS API SEARCH
    url = f"https://newsapi.org/v2/everything?q={full_text}&apiKey={API_KEY}"

    response = requests.get(url)

    news_data = response.json()

    articles = []

    for article in news_data.get("articles", [])[:5]:

        articles.append({
            "title": article.get("title", ""),
            "url": article.get("url", "")
        })

    # FINAL RESPONSE
    return jsonify({
        "prediction": result,
        "confidence": confidence,
        "articles": articles,
        "ocr_text": extracted_text
    })

    import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
