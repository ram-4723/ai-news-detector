from flask import Flask, render_template, request, jsonify
import pickle
import requests
import os

app = Flask(__name__)

# API KEYS
API_KEY = "2d032abfb1e249bca4299081e978a769"
OCR_API_KEY = "K85609959888957"

# LOAD ML MODEL
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))


# HOME PAGE
@app.route("/")
def home():
    return render_template("index.html")


# AI ANALYSIS API
@app.route("/analyze", methods=["POST"])
def analyze():

    try:

        # GET TEXT + IMAGE
        news = request.form.get("news", "")
        image = request.files.get("image")

        extracted_text = ""

        # OCR IMAGE EXTRACTION USING OCR.SPACE API
        if image:

            try:

                response = requests.post(
                    "https://api.ocr.space/parse/image",
                    files={"filename": image},
                    data={
                        "apikey": OCR_API_KEY,
                        "language": "eng"
                    }
                )

                result = response.json()

                if (
                    "ParsedResults" in result
                    and len(result["ParsedResults"]) > 0
                ):
                    extracted_text = result["ParsedResults"][0].get("ParsedText", "")

            except Exception as ocr_error:
                print("OCR ERROR:", ocr_error)
                extracted_text = ""

        # COMBINE NEWS + OCR TEXT
        full_text = (news + " " + extracted_text).strip()

        # VALIDATION
        if len(full_text.split()) < 3:

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
        articles = []

        try:

            url = f"https://newsapi.org/v2/everything?q={full_text}&apiKey={API_KEY}"

            response = requests.get(url)

            news_data = response.json()

            for article in news_data.get("articles", [])[:5]:

                articles.append({
                    "title": article.get("title", ""),
                    "url": article.get("url", "")
                })

        except Exception as news_error:
            print("NEWS API ERROR:", news_error)

        # FINAL RESPONSE
        return jsonify({
            "prediction": result,
            "confidence": confidence,
            "articles": articles,
            "ocr_text": extracted_text
        })

    except Exception as e:

        print("SERVER ERROR:", e)

        return jsonify({
            "prediction": "Server Error",
            "confidence": 0,
            "articles": [],
            "ocr_text": ""
        }), 500


# RUN APP
if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(host="0.0.0.0", port=port)
