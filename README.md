# Lumina

**Illuminating Student Learning Through Deep Knowledge Tracing**

Lumina is an end-to-end Deep Knowledge Tracing (DKT) system that predicts student performance on educational questions and surfaces actionable insights through an interactive instructor dashboard. Built on an LSTM-based architecture trained on the XES3G5M dataset, Lumina gives educators real-time visibility into where every student stands --- enabling earlier interventions, targeted remediation, and more equitable learning outcomes.

---

## Table of Contents

- [Setup](#setup)
  - [Backend](#backend)
  - [Frontend](#frontend)
  - [Model Training](#model-training)
- [Architecture Overview](#architecture-overview)
- [Model](#model)
  - [DKTLSTM Architecture](#dktlstm-architecture)
  - [Input Features](#input-features)
  - [Training](#training)
- [Dataset](#dataset)
- [Dashboard](#dashboard)
- [API Reference](#api-reference)
- [Impact on Learning Outcomes](#impact-on-learning-outcomes)

---

## Setup

### Prerequisites

- Python 3.10+
- Node.js 18+
- npm or yarn

### Backend

```bash
# Create and activate a virtual environment
python3 -m venv backend/.venv
source backend/.venv/bin/activate

# Install dependencies
python3 -m pip install --upgrade pip
python3 -m pip install -r backend/requirements.txt

# Start the API server
uvicorn backend.main:app --reload
```

The backend will be available at `http://127.0.0.1:8000`.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

The dashboard will be available at `http://localhost:3000`. The frontend expects the backend to be running --- configure a custom backend URL via the `BACKEND_API_URL` or `NEXT_PUBLIC_BACKEND_API_URL` environment variable if needed.

### Running without training

If you clone the repo and **do not** want to train the model (e.g. your machine is too slow), you can still run the web app using pre-trained weights:

1. **Get the pre-trained weight files** from someone who has already run training, or from a shared drive. You need these 10 files placed in `students/student_weights/`:
   - `student_1_memory_state.pt`, `student_1_model_with_memory.pt`
   - `student_2_memory_state.pt`, `student_2_model_with_memory.pt`
   - `student_3_memory_state.pt`, `student_3_model_with_memory.pt`
   - `student_4_memory_state.pt`, `student_4_model_with_memory.pt`
   - `student_5_memory_state.pt`, `student_5_model_with_memory.pt`

2. The repo already includes `students/student_weights/summary.json` and the student CSVs (`student_1.csv` … `student_5.csv`). Once the `.pt` files above are in `students/student_weights/`, start the [backend](#backend) and [frontend](#frontend) as usual — no training or dataset is required.

### Model Training

Training is only required if you want to retrain from scratch; pre-trained weights are included in `students/student_weights/` when provided (see [Running without training](#running-without-training)).

```bash
cd model-training

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

**Run the data preprocessing pipeline:**

```bash
python3 -m src.data.run_qlevel_pipeline \
  --raw-root XES3G5M \
  --output-root data \
  --window-size 200 \
  --stride 50 \
  --min-user-history 3
```

**Train the model (single fold or full cross-validation):**

```bash
# Single fold
python3 -m src.modeling.train_dkt_lstm \
  --data-dir data/processed/dkt_qlevel_v1 \
  --output-dir artifacts/dkt_lstm \
  --run-name experiment \
  --fold 0 \
  --device mps   # use "cuda" for NVIDIA GPUs or "cpu"

# 5-fold cross-validation
python3 -m src.modeling.train_dkt_lstm \
  --data-dir data/processed/dkt_qlevel_v1 \
  --output-dir artifacts/dkt_lstm \
  --run-name cv_experiment \
  --all-folds \
  --device mps
```

---

## Architecture Overview

Lumina is composed of three layers:

| Layer | Technology | Role |
|---|---|---|
| **Model Training** | PyTorch, scikit-learn | Trains the DKTLSTM model on preprocessed XES3G5M sequences and exports per-student checkpoints |
| **Backend API** | FastAPI, pandas, NumPy | Loads model weights, runs inference on student interaction histories, and computes analytics |
| **Frontend Dashboard** | Next.js 16, React 19, Tailwind CSS, Recharts | Renders an interactive instructor dashboard with tables, heatmaps, and distribution charts |

Data flows from raw CSV interaction logs through a 7-stage preprocessing pipeline, into LSTM training, and finally to the dashboard via a REST API.

---

## Model

### DKTLSTM Architecture

The core model (`DKTLSTM`) is a multi-input LSTM that extends the standard Deep Knowledge Tracing formulation with concept tags, numeric difficulty features, and semantic embeddings.

```
┌─────────────────────────────────────────────────────────────────┐
│                         DKTLSTM                                 │
│                                                                 │
│  Interaction Embedding ──┐                                      │
│  (question × response)   │                                      │
│                          │                                      │
│  Concept Embedding ──────┤  (mean-pooled over tags per step)    │
│                          │                                      │
│  Numeric Features ───────┤  delta_t embed + q_type embed        │
│  (7 scalars → Linear)    │  + linear projection                 │
│                          │                                      │
│  Semantic Vectors ───────┤  content + analysis embeddings       │
│  (2 × dim + 2 → Linear) │  + missing indicators                │
│                          │                                      │
│                     concat → LSTM → Dropout → Linear(num_q)     │
│                                                                 │
│  Loss: Masked BCEWithLogits on next-question correctness        │
└─────────────────────────────────────────────────────────────────┘
```

**Default hyperparameters:**

| Parameter | Value |
|---|---|
| Interaction embedding dim | 64 |
| LSTM hidden size | 128 |
| LSTM layers | 1 |
| Dropout | 0.2 |
| Concept embedding dim | 32 |
| Delta-t embedding dim | 8 |
| Question-type embedding dim | 4 |
| Numeric projection dim | 16 |
| Semantic projection dim | 32 |
| Semantic vector dim | 32 |

### Input Features

Each timestep in a student's interaction sequence is represented by the following features:

| Feature | Type | Description |
|---|---|---|
| `interaction_ids` | int | Encoded question-response pair: `question_idx + response × num_questions` |
| `concept_indices` | int (padded) | Knowledge-component tags associated with the question |
| `concept_mask` | float | Mask indicating valid concepts at each timestep |
| `delta_t_log_bin` | int (0--20) | Log-binned time elapsed since the previous interaction |
| `question_type` | int (0--2) | Category of the question |
| `num_concepts` | float | Number of knowledge components tagged to the question |
| `concept_rarity_bucket` | float | How rare the associated concepts are across the dataset |
| `hist_accuracy` | float | Student's running historical accuracy up to this point |
| `question_difficulty_prior` | float | Smoothed prior difficulty of the question |
| `concept_difficulty_prior` | float | Smoothed prior difficulty at the concept level |
| `content_vectors` | float (dim 32) | Semantic embedding of the question content |
| `analysis_vectors` | float (dim 32) | Semantic embedding of the question analysis/solution |
| `content_embedding_missing` | float | Indicator for whether the content embedding is present |
| `analysis_embedding_missing` | float | Indicator for whether the analysis embedding is present |

### Training

- **Loss function:** Masked Binary Cross-Entropy with Logits --- only supervised on timesteps where a valid next-question response exists.
- **Evaluation metric:** AUC-ROC, computed on masked predictions.
- **Cross-validation:** 5-fold, using the fold splits provided in the XES3G5M dataset.
- **Windowing:** Sequences are sliced into overlapping windows of 200 timesteps with a stride of 50. Students with fewer than 3 interactions are excluded.

---

## Dataset

Lumina is trained on **XES3G5M**, a large-scale question-level knowledge tracing dataset.

**Key statistics:**

- Covers K-12 mathematics across 1,174 distinct knowledge concepts (mapped to English labels such as *Perimeter of Rectangle*, *Route Matching Problem*, *Find Contradictions*, etc.)
- Interaction sequences contain question IDs, binary correctness responses, timestamps, and associated concept tags
- Provided with 5-fold cross-validation splits

**Preprocessing pipeline** (`src.data.run_qlevel_pipeline`):

1. **Ingest** raw CSV sequences into a long-format interaction table
2. **Clean & validate** --- remove malformed rows, filter short histories
3. **Build features** --- compute difficulty priors, rarity buckets, delta-t bins, historical accuracy, and semantic embeddings per fold
4. **Package sequences** into overlapping JSONL windows
5. **Run quality checks & ablation reports**
6. **Validate** final output structure

Processed data is stored under `data/processed/dkt_qlevel_v1/` with per-fold train/validation splits, a `feature_state.json` vocabulary file, and `semantic_embedding_tables.json` for precomputed question embeddings.

---

## Dashboard

The Lumina dashboard is an interactive Next.js application that lets instructors explore model predictions and student analytics at a glance.

**Key views:**

- **Summary cards** --- Class average, median score, percentage of students below threshold, hardest/easiest questions, and completion rate
- **Question analysis table** --- Per-question accuracy, attempt counts, skill tags, and flagging for questions below a configurable difficulty threshold
- **Student performance table** --- Per-student score, completion status, time spent, median spacing between attempts, and recommended areas to study
- **Student x Question matrix** --- Color-coded heatmap showing correct / incorrect / unanswered states alongside model-predicted probabilities for unanswered questions
- **Concept accuracy breakdown** --- Aggregated accuracy by knowledge component across the class
- **Score distribution chart** --- Histogram of student scores
- **Remediation panel** --- Auto-generates a targeted question set for flagged students, showing skill coverage and estimated time

Instructors can adjust the student and question thresholds to dynamically re-flag at-risk students and difficult questions, and can select specific test questions to run live model inference across all students.

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check --- returns `{"status": "ok"}` |
| `POST` | `/predict` | Upload a student CSV and receive per-question correctness probabilities |
| `GET` | `/dashboard-data` | Retrieve the full dashboard payload (students, questions, matrix, remediation) |
| `POST` | `/process-selected-questions` | Run model inference on a chosen set of test questions for all students |

**`/dashboard-data` query parameters:**

| Parameter | Default | Description |
|---|---|---|
| `student_threshold` | 70.0 | Score percentage below which a student is flagged |
| `question_threshold` | 60.0 | Accuracy percentage below which a question is flagged |
| `exclude_unanswered` | false | Whether to exclude unanswered questions from stats |
| `top_n_hardest` | 15 | Number of hardest questions to include |

---

## Impact on Learning Outcomes

Lumina is built with a conviction that the right information at the right time can fundamentally change a student's trajectory. By giving instructors a continuous, data-driven understanding of where every student stands, Lumina unlocks several high-impact opportunities for improving education.

### Early Intervention

Traditional assessments surface struggles only after the fact --- after a midterm is graded, after a homework deadline passes. Lumina's predictive model estimates the probability that each student will answer each question correctly *before* they attempt it. Instructors can identify students who are beginning to fall behind while there is still time to act, reaching out with targeted support before small gaps compound into lasting knowledge deficits.

### Identifying Students Who Need Extra Support

The dashboard automatically flags students whose predicted or actual scores fall below a configurable threshold. Rather than relying on students to self-report difficulty or waiting for failing grades to appear, instructors gain a persistent early-warning system. This is especially important for students who may be reluctant to ask for help, ensuring that no one slips through the cracks.

### Targeted Remediation

Lumina does not just flag problems --- it pinpoints *what* to work on. The concept-level accuracy breakdown and auto-generated remediation plans show exactly which knowledge components a struggling student has not yet mastered. Instructors can assign focused practice on the specific skills that will have the greatest impact, rather than prescribing one-size-fits-all review.

### Equitable Access to Attention

In large classrooms, instructor attention is a scarce resource. Without data, it tends to flow toward the most vocal students or the most obviously struggling ones. Lumina democratizes this attention by making every student's learning state visible. Quiet students who are silently falling behind receive the same chance at support as those who actively seek it --- working toward a more equitable classroom.

### Understanding Question Quality

The question analysis table reveals which questions are disproportionately difficult or poorly understood. Instructors can use this feedback loop to improve their assessments: rewriting confusing questions, adjusting difficulty curves, and ensuring that assessments genuinely measure the intended learning objectives rather than incidental confusion.

### Data-Driven Curriculum Design

Aggregated concept-level analytics show where an entire class is struggling. If the majority of students have low predicted accuracy on a particular set of knowledge components, that is a signal that the curriculum, pacing, or instructional approach for those topics may need revision. Over multiple terms, this feedback enables continuous improvement of course design grounded in evidence rather than intuition.

### Reducing Achievement Gaps

Students from disadvantaged backgrounds are disproportionately affected when struggles go unnoticed. By making learning gaps visible early and systematically, Lumina helps instructors direct resources where they are most needed --- a concrete step toward closing achievement gaps and advancing educational equity as a matter of social good.

### Supporting Student Agency

When prediction data is shared with students themselves, it can foster metacognitive awareness. Students can see which areas the model identifies as their weakest, guiding their own study plans and building self-regulated learning habits that persist well beyond any single course.

---

## License

This project was developed for the UCSB Datathon 2026.
