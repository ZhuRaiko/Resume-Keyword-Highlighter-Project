# Streamlit Deployment Guide

This project is a Streamlit app, so the recommended deployment target is Streamlit Community Cloud rather than Vercel.

## Why Vercel Failed

Vercel is mainly optimized for frontend frameworks and serverless functions. This project runs as a long-lived Streamlit Python app, which expects Streamlit's own server process.

Use Hugging Face Spaces, Streamlit Community Cloud, Render, Railway, or another Python app host instead.

## Recommended Alternative: Hugging Face Spaces

Hugging Face Spaces is a good fit for this project because it is an ML demo app. For Streamlit apps, use a Docker Space.

### Create The Space

1. Go to Hugging Face.
2. Open **Spaces**.
3. Click **Create new Space**.
4. Choose a Space name.
5. Choose visibility: Public or Private.
6. Choose **Docker** as the SDK.
7. Create the Space.

### Use This Repo

You have two common options:

1. Upload or push this repository's files into the Space repository.
2. Connect/sync from GitHub if you prefer managing the source in GitHub.

The important files for Hugging Face Spaces are:

```text
Dockerfile
.dockerignore
requirements.txt
main_modular.py
.streamlit/config.toml
```

The Dockerfile runs:

```bash
streamlit run main_modular.py --server.address=0.0.0.0 --server.port=8501 --server.enableXsrfProtection=false --server.enableCORS=false
```

The XSRF/CORS flags are included because this app uses file upload, and Streamlit file upload can be affected by iframe/cookie restrictions on Hugging Face Spaces when using Docker.

### Hugging Face App Port

The Dockerfile exposes port `8501`, which is Streamlit's default port.

If Hugging Face asks for an app port in Space settings, use:

```text
8501
```

### First Build

The first Hugging Face build can take a while because it installs NLP and ML dependencies. Later rebuilds should be easier unless dependency files change.

## Recommended Deployment: Streamlit Community Cloud

1. Push the repository to GitHub.
2. Go to Streamlit Community Cloud.
3. Create a new app.
4. Select the GitHub repository.
5. Select the branch, usually `main`.
6. Set the main file path to:

```text
main_modular.py
```

7. Deploy the app.

## Files Needed For Deployment

The important deployment files are:

```text
requirements.txt
.python-version
.streamlit/config.toml
main_modular.py
```

`requirements.txt` contains the app dependencies that Streamlit Cloud installs.

`.python-version` requests Python 3.11.

`.streamlit/config.toml` contains the dark theme and headless server settings.

## Local Test Command

Run this from the project root:

```bash
streamlit run main_modular.py
```

Do not run the app with:

```bash
python main_modular.py
```

Using plain Python can trigger this warning:

```text
Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
```

That warning means Streamlit code was executed outside Streamlit's normal runtime context.

## Dependency Notes

The root `requirements.txt` is intentionally focused on the app, not the full evaluation environment.

It includes:

- Streamlit UI dependencies.
- SentenceTransformer and spaCy for NLP.
- The `en_core_web_sm` spaCy model from its wheel URL.
- scikit-learn and joblib for the cached KNN model.
- PDF/DOCX/TXT extraction dependencies.

The evaluation-only packages and plotting packages are not included because Streamlit Cloud only needs to run the app.

## If Deployment Fails

Check these first:

1. The main file path is exactly `main_modular.py`.
2. `requirements.txt` is in the repository root.
3. `models/knn_model.pkl` is committed.
4. `data/keywords.json` is committed.
5. `data/self_promotion_dataset.csv` is committed.
6. The font files in `fonts/` are committed.

If the app fails while loading spaCy, check whether the `en_core_web_sm` line in `requirements.txt` installed successfully.

If the app fails while processing PDFs, try DOCX or pasted text first. PDF extraction depends heavily on the PDF's formatting.
