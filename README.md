# CareerLens AI

CareerLens AI is a Vercel-ready Flask machine-learning application that estimates a student's placement readiness score and returns focused improvement recommendations.

## Features

- Placement readiness score from 0–100
- Four readiness bands
- Prioritised personalised action plan
- Model feature-importance view
- Responsive original frontend
- JSON prediction and health endpoints
- Reproducible training script and transparent synthetic dataset

## Repository structure

```text
careerlens-vercel/
├── app.py
├── train_model.py
├── requirements.txt
├── vercel.json
├── README.md
├── LICENSE
├── .gitignore
├── data/
│   └── career_readiness_synthetic.csv
├── model/
│   └── careerlens_model.pkl
├── static/
│   ├── app.js
│   └── style.css
└── templates/
    └── index.html
```

## Run locally

Python 3.12 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000`.

The trained model is already included. Run `python train_model.py` only when you want to reproduce the dataset and model.

## Deploy on Vercel

1. Create an empty GitHub repository.
2. Upload **the contents inside this folder**, keeping the same directories.
3. In Vercel, choose **Add New → Project** and import that repository.
4. Keep Framework Preset as detected (Flask) and Root Directory as `./`.
5. Do not add a Build Command, Output Directory, or environment variables.
6. Select **Deploy**.

The included `vercel.json` declares the Flask framework. Vercel detects `app.py` and installs packages from `requirements.txt`.

## API

Health check:

```http
GET /api/health
```

Prediction:

```http
POST /api/predict
Content-Type: application/json
```

Use the field names shown in `FEATURES` inside `app.py`.

## Model note

The included training data is synthetic and generated with a documented scoring process in `train_model.py`. This makes the demo reproducible and avoids presenting private or invented survey data as real research. For academic publication or production hiring decisions, replace it with a properly collected, consented and validated dataset.

## License

MIT

