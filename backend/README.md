# Student Prediction API

Minimal FastAPI service that accepts an enriched student CSV and returns per-row predicted correctness probabilities.

## Setup

From repo root:

```bash
python3 -m venv backend/.venv
source backend/.venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r backend/requirements.txt
```

If you already use `model-training/.venv` with torch installed, you can install only API deps there and run from that environment.

## Run

From repo root:

```bash
uvicorn backend.main:app --reload
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

## Predict

Example request using student-specific pretrained artifact from `students/student_weights/`:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -F "student_id=1" \
  -F "file=@students/student_1.csv;type=text/csv"
```

Response shape:

```json
{
  "student_id": 1,
  "user_id": 13107,
  "probabilities": [0.12, 0.09, 0.07]
}
```

## Smoke test

```bash
python3 backend/tests/smoke_test.py
```

