# AI Fake News Detector

A small Flask web application that classifies submitted news text with the included TF-IDF and Naive Bayes model. It can extract text from an uploaded image with Tesseract OCR and optionally search NewsAPI for related articles.

## Deploy to Render

This repository includes a Render Blueprint (`render.yaml`) and a Dockerfile. Docker installs the Tesseract system executable as well as the Python packages, so OCR does not depend on packages present in Render's native Python runtime.

1. Push the project files to the GitHub repository. Keep `model.pkl`, `vectorizer.pkl`, `Dockerfile`, and `render.yaml` in the repository root.
2. In Render, select **New → Blueprint**, connect the repository, and apply the Blueprint. Alternatively, create a **Web Service** from the repository and select **Docker** as the runtime.
3. In the service's **Environment** settings, add `NEWSAPI_KEY` with a newly issued NewsAPI key. The key is optional; without it, text/image analysis works and the related-articles list is empty.
4. Deploy. Render checks `/health` and starts the site at the service URL.

Do not put API keys in source code or commit them. If a key was previously committed, revoke it at its provider and create a replacement.

## Local run

Requires Python 3.12, Tesseract OCR installed on the machine, and the dependencies in `requirements.txt`.

```sh
python -m pip install -r requirements.txt
gunicorn --bind 0.0.0.0:5000 app:app
```

Open `http://localhost:5000`. Set `NEWSAPI_KEY` in the environment if related-article lookup is desired. On Windows, set `TESSERACT_CMD` to the full path of `tesseract.exe` if it is not on `PATH`.

## Notes

- The committed model files were serialized with scikit-learn 1.8.0; the dependency is pinned to that version for compatibility.
- The confidence value is the model's class probability, not a fact-check guarantee. Check original sources before treating a result as true or false.
- `train.py` is an optional training script and expects `Fake.csv` and `True.csv`, which are not included in this repository. Training is not part of deployment.
