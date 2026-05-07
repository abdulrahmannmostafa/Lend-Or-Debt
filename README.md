# Lend or Debt

A full-stack data science project for **credit default risk analysis** on the Taiwan credit card dataset, with:

- a reproducible Python data pipeline,
- model training and experiment tracking with MLflow,
- a Flask API to run pipeline stages and serve EDA images,
- and a React dashboard for interactive exploration.

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Tech Stack](#tech-stack)
- [Pipeline Stages](#pipeline-stages)
- [Model Training and Selection](#model-training-and-selection)
- [Backend API](#backend-api)
- [Frontend](#frontend)
- [Getting Started](#getting-started)
- [Development Commands](#development-commands)
- [Environment Variables](#environment-variables)
- [Deployment](#deployment)

## Overview

`Lend-Or-Debt` processes credit and macroeconomic data, validates dataset quality, prepares training features, and trains multiple classifiers to predict **`default payment next month`**.

The project is organized as a multi-phase pipeline:

1. Data acquisition and source merging
2. Dataset validation (rule/statistical checks implemented in Python)
3. Cleaning and train/validation/test split
4. Feature transformation and SMOTE/no-SMOTE outputs
5. EDA generation
6. Model training + MLflow tracking

## Repository Structure

```text
Lend-Or-Debt/
├── src/
│   ├── pipeline/
│   │   ├── config.py
│   │   ├── data_acquisition.py
│   │   ├── data_validation.py
│   │   ├── data_cleaning.py
│   │   ├── data_transformation.py
│   │   ├── eda.py
│   │   ├── feature_extraction.py
│   │   ├── master_pipeline.py
│   │   └── mlflow/
│   │       ├── model_selection_evaluations.py
│   │       ├── models.py
│   │       └── ...
│   ├── scrape/
│   │   ├── fetch_fred_api.py
│   │   ├── merge_sources.py
│   │   ├── parse_cbc_pdf.py
│   │   ├── scrape_taiex.py
│   │   └── scrape_unemployment.py
│   └── notebooks/
├── backend/
│   ├── app.py
│   ├── requirements.txt
│   ├── Procfile
│   └── railway.json
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── api.js
│   │   └── App.jsx
│   ├── package.json
│   └── vercel.json
├── data/
│   ├── clean/
│   └── transformed/
├── tests/
│   ├── unit/
│   └── integration/
├── Makefile
├── pyproject.toml
└── README.md
```

## Tech Stack

### Core (Python)
- Python 3.11+
- Poetry
- pandas, NumPy, SciPy
- scikit-learn, imbalanced-learn
- xgboost, lightgbm, catboost
- mlflow, optuna
- matplotlib, seaborn, plotly
- Selenium, BeautifulSoup, requests, fredapi, yfinance

### Backend
- Flask
- Flask-CORS

### Frontend
- React
- Vite

### Quality
- Ruff
- mypy
- pytest

## Pipeline Stages

The pipeline is orchestrated by `src/pipeline/master_pipeline.py` and can be run end-to-end or phase-by-phase.

| Stage | Module | Main Output |
|---|---|---|
| 1. Acquisition | `src/pipeline/data_acquisition.py` | `data/taiwan_merged.csv` |
| 2. Validation | `src/pipeline/data_validation.py` | `data/validation_results.json`, `data/validation_report.md` |
| 3. Cleaning | `src/pipeline/data_cleaning.py` | `data/clean/*.csv` |
| 4. Transformation | `src/pipeline/data_transformation.py` | `data/transformed/*.csv` |
| 5. EDA | `src/pipeline/eda.py` | Generated visual outputs |
| 6. Modeling | `src/pipeline/mlflow/model_selection_evaluations.py` | MLflow runs + metrics |

Run all phases:

```bash
make pipeline
```

Resume from a specific phase:

```bash
poetry run python -m src.pipeline.master_pipeline --from 3
```

## Model Training and Selection

Run the training/selection command:

```bash
make modeling_selection model=<id> smote=<0|1> fs=<spearman|mutual_info|intersect> ver=<n> k=<n>
```

Model IDs:

| ID | Model |
|---|---|
| 0 | Logistic Regression |
| 1 | Random Forest |
| 2 | XGBoost |
| 3 | LightGBM |
| 4 | Extra Trees |
| 5 | CatBoost |
| 6 | Decision Tree |
| 7 | AdaBoost |
| 8 | KNN |
| 9 | SVM |

MLflow experiment name used in code: `data_science_Project_team1`.

## Backend API

The Flask backend in `backend/app.py` exposes endpoints under `/api`:

- `POST /upload`
- `POST /run/cleaning`
- `POST /run/transformation`
- `POST /eda/init`
- `GET /eda/columns`
- `POST /eda/univariate`
- `POST /eda/pie`
- `POST /eda/continuous`
- `POST /eda/discrete_vs_continuous`
- `POST /eda/discrete_vs_target`
- `POST /eda/correlation`
- `POST /eda/dashboard_with_smote`
- `POST /eda/dashboard_without_smote`
- `GET /debug`

Most EDA endpoints return a base64 PNG image in JSON:

```json
{ "image": "<base64...>" }
```

## Frontend

The React app (`frontend/`) supports:

- uploading a CSV,
- triggering cleaning/transformation,
- initializing EDA,
- rendering EDA images from API responses.

Frontend API base URL is read from:

- `VITE_API_URL` (if set), otherwise
- `http://localhost:5000/api`

## Getting Started

### Prerequisites

- Python 3.11+
- Poetry
- Node.js 18+
- FRED API key

### 1) Clone

```bash
git clone https://github.com/abdulrahmannmostafa/Lend-Or-Debt.git
cd Lend-Or-Debt
```

### 2) Install Python dependencies

```bash
poetry install
```

### 3) Configure environment

```bash
cp .env.example .env
# Set FRED_API_KEY in .env
```

### 4) Run pipeline

```bash
make pipeline
```

### 5) Run backend

```bash
cd backend
pip install -r requirements.txt
python app.py
```

### 6) Run frontend (new terminal)

```bash
cd frontend
npm install
npm run dev
```

## Development Commands

From repository root:

```bash
make lint
make format
make sta
make unit
make integration
```

## Environment Variables

| Variable | Description |
|---|---|
| `FRED_API_KEY` | Key for FRED macroeconomic API access |
| `VITE_API_URL` | Frontend base URL for backend API (frontend runtime) |

## Deployment

- Backend: Railway (`backend/Procfile`, `backend/railway.json`)
- Frontend: Vercel (`frontend/vercel.json`)
