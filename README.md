# AI Fake News Detector

A Flask application that requires NewsAPI to search for articles related to submitted text. It extracts text from uploaded images with Tesseract OCR and shows matching article links as evidence for the user to review. A saved TF-IDF/Naive Bayes model is shown only as a secondary estimate.

## Deploy to Render

This repository includes a Render Blueprint (`render.yaml`) and a Dockerfile. Docker installs the Tesseract system executable as well as the Python packages, so OCR does not depend on packages present in Render's native Python runtime.

1. Push the project files to the GitHub repository. Keep `model.pkl`, `vectorizer.pkl`, `Dockerfile`, and `render.yaml` in the repository root.
2. In Render, select **New → Blueprint**, connect the repository, and apply the Blueprint. Alternatively, create a **Web Service** from the repository and select **Docker** as the runtime.
3. In the service's **Environment** settings, add `NEWSAPI_KEY` with a valid NewsAPI key. This is required: without it (or if NewsAPI is unavailable), analysis returns an error instead of showing a model-only result.
4. Deploy. Render checks `/health` and starts the site at the service URL.

Do not put API keys in source code or commit them. If a key was previously committed, revoke it at its provider and create a replacement.

**NewsAPI plan warning:** NewsAPI's free Developer plan is limited to development and testing and cannot be used for a production website. A live public deployment needs an eligible paid plan. Check NewsAPI's current [pricing and usage terms](https://newsapi.org/pricing) before putting the key on Render.

## Local run

Requires Python 3.12, Tesseract OCR installed on the machine, and the dependencies in `requirements.txt`.

```sh
python -m pip install -r requirements.txt
gunicorn --bind 0.0.0.0:5000 app:app
```

Open `http://localhost:5000`. Set `NEWSAPI_KEY` in the environment before using analysis. On Windows, set `TESSERACT_CMD` to the full path of `tesseract.exe` if it is not on `PATH`.

## Notes

- The committed model files were serialized with scikit-learn 1.8.0; the dependency is pinned to that version for compatibility.
- NewsAPI search is article discovery, not an automated fact-check. Headlines and descriptions may be incomplete or irrelevant; the app therefore reports an inconclusive evidence result and links to sources for manual review.
- The optional model probability is not a fact-check guarantee. Check original sources before treating a claim as true or false.
- `train.py` is an optional training script and expects `Fake.csv` and `True.csv`, which are not included in this repository. Training is not part of deployment.
