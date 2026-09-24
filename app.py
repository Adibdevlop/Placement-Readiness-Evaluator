from pathlib import Path

import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "careerlens_model.pkl"

app = Flask(__name__)
bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
feature_names = bundle["feature_names"]


FEATURES = {
    "cgpa": (0.0, 10.0),
    "dsa_questions": (0, 1000),
    "coding_skill": (1, 10),
    "aptitude_score": (0, 100),
    "communication_skill": (1, 10),
    "projects": (0, 15),
    "internships": (0, 8),
    "mock_interviews": (0, 50),
    "resume_score": (0, 100),
    "weekly_prep_hours": (0, 80),
}


def validate_payload(payload):
    values = []
    errors = {}
    for name in feature_names:
        try:
            value = float(payload.get(name, ""))
            low, high = FEATURES[name]
            if not low <= value <= high:
                errors[name] = f"Enter a value between {low} and {high}."
            values.append(value)
        except (TypeError, ValueError):
            errors[name] = "Enter a valid number."
            values.append(0.0)
    return values, errors


def readiness_level(score):
    if score >= 80:
        return "Placement Ready", "excellent"
    if score >= 65:
        return "Nearly Ready", "good"
    if score >= 45:
        return "Developing", "medium"
    return "Foundation Stage", "low"


def recommendations(data):
    checks = [
        (data["coding_skill"] < 7, "Coding", "Practise timed coding and revise core language concepts."),
        (data["dsa_questions"] < 200, "DSA", "Build toward 200 quality problems with revision notes."),
        (data["communication_skill"] < 7, "Communication", "Record two-minute project explanations and review them."),
        (data["projects"] < 2, "Projects", "Complete one deployable project with a strong README and live demo."),
        (data["internships"] < 1, "Experience", "Apply to internships or contribute to a real open-source issue."),
        (data["mock_interviews"] < 5, "Interviews", "Schedule at least five technical and HR mock interviews."),
        (data["resume_score"] < 75, "Resume", "Quantify project impact and keep the resume to one focused page."),
        (data["aptitude_score"] < 70, "Aptitude", "Take topic-wise timed tests and maintain an error log."),
        (data["weekly_prep_hours"] < 12, "Consistency", "Create a weekly plan with at least 12 focused preparation hours."),
    ]
    result = [{"area": area, "action": action} for failed, area, action in checks if failed]
    return result[:4] or [{"area": "Momentum", "action": "Maintain your routine and target company-specific mock tests."}]


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "model": "CareerLens Random Forest Regressor"})


@app.post("/api/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    values, errors = validate_payload(payload)
    if errors:
        return jsonify({"errors": errors}), 400

    score = float(np.clip(model.predict([values])[0], 0, 100))
    rounded_score = round(score, 1)
    level, tone = readiness_level(rounded_score)
    clean_data = dict(zip(feature_names, values))

    importance = sorted(
        zip(feature_names, model.feature_importances_),
        key=lambda item: item[1],
        reverse=True,
    )[:5]

    return jsonify({
        "score": rounded_score,
        "level": level,
        "tone": tone,
        "recommendations": recommendations(clean_data),
        "top_factors": [
            {"name": name.replace("_", " ").title(), "importance": round(float(value) * 100, 1)}
            for name, value in importance
        ],
        "disclaimer": "This result is an educational estimate, not a hiring guarantee.",
    })


if __name__ == "__main__":
    app.run(debug=True)

