# Lend or Debt

> **End-to-end credit card default prediction** — from raw data acquisition through model selection, with an interactive React dashboard powered by a Flask REST API.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Pipeline Stages](#pipeline-stages)
- [Model Training & Selection](#model-training--selection)
- [Backend API](#backend-api)
- [Frontend Dashboard](#frontend-dashboard)
- [Development Commands](#development-commands)
- [Environment Variables](#environment-variables)
- [Deployment](#deployment)

---

## Overview

**Lend or Debt** predicts whether a credit card client will default on their next payment. It combines a curated Taiwan credit card dataset with macroeconomic signals (interest rates, unemployment, stock index) sourced from public APIs and web scraping.

The system is built as a **fully reproducible, phased pipeline**:

1. Acquire and merge data from multiple sources
2. Validate data quality with Pandas
3. Clean and split into train / validation / test sets
4. Transform features (encoding, scaling, SMOTE oversampling)
5. Perform exploratory data analysis (EDA)
6. Train and track multiple ML models via MLflow

Results are served through a **Flask REST API** and visualised in a **React + Vite** interactive dashboard.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        ML Pipeline (Python)                      │
│  Acquisition → Validation → Cleaning → Transformation → EDA     │
│                          ↓ MLflow                                │
│              Model Training & Selection (10 models)             │
└──────────────────────────────┬───────────────────────────────────┘
                               │ REST API
               ┌───────────────▼────────────────┐
               │        Flask Backend            │
               │  (Railway)  port 5000           │
               └───────────────┬────────────────┘
                               │ HTTP
               ┌───────────────▼────────────────┐
               │      React + Vite Frontend      │
               │         (Vercel)                │
               └────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python ≥ 3.11 |
| Dependency management | [Poetry](https://python-poetry.org/) |
| Data processing | pandas, scikit-learn, imbalanced-learn |
| Data validation | [Great Expectations](https://greatexpectations.io/) |
| Experiment tracking | [MLflow](https://mlflow.org/) |
| Hyperparameter tuning | [Optuna](https://optuna.org/) |
| ML models | XGBoost, LightGBM, CatBoost, Random Forest, Extra Trees, AdaBoost, Logistic Regression, KNN, SVM, Decision Tree |
| Web scraping | Selenium, BeautifulSoup4, requests |
| Macro data | [FRED API](https://fred.stlouisfed.org/docs/api/fred/) (`fredapi`), `yfinance` |
| Backend | Flask + Flask-CORS |
| Frontend | React 19, Vite 8 |
| Linting / formatting | Ruff, mypy |
| Testing | pytest, pytest-cov |
| Logging | Loguru |

---

## Project Structure

```
Lend-Or-Debt/
├── src/
│   ├── pipeline/
│   │   ├── config.py                   # Centralised path constants
│   │   ├── data_acquisition.py         # Phase 1 — fetch & merge sources
│   │   ├── data_validation.py          # Phase 2 — Great Expectations checks
│   │   ├── data_cleaning.py            # Phase 3 — clean & train/val/test split
│   │   ├── data_transformation.py      # Phase 4 — encoding, scaling, SMOTE
│   │   ├── eda.py                      # Phase 5 — EDA plots & dashboards
│   │   ├── feature_extraction.py       # Feature selection (Spearman / MI)
│   │   ├── master_pipeline.py          # Orchestrates all phases end-to-end
│   │   └── mlflow/
│   │       ├── models.py               # Model class definitions
│   │       ├── model_selection_evaluations.py  # Phase 6 — training & tracking
│   │       ├── create_environment.py
│   │       ├── delete_exp.py
│   │       ├── globals.py
│   │       └── retrieve_exp.py
│   └── scrape/
│       ├── fetch_fred_api.py           # FRED macroeconomic series
│       ├── merge_sources.py            # Merge all raw sources
│       ├── parse_cbc_pdf.py            # CBC annual report PDF parser
│       ├── scrape_taiex.py             # TAIEX stock index scraper
│       └── scrape_unemployment.py      # Unemployment rate scraper
├── backend/
│   ├── app.py                          # Flask REST API
│   ├── requirements.txt
│   ├── Procfile                        # Railway process file
│   └── railway.json
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── EDASection.jsx
│   │   │   ├── Intro.jsx
│   │   │   ├── LogBox.jsx
│   │   │   ├── PlotImg.jsx
│   │   │   ├── RunPhase.jsx
│   │   │   ├── Spinner.jsx
│   │   │   └── UploadSection.jsx
│   │   ├── App.jsx
│   │   ├── api.js
│   │   ├── main.jsx
│   │   └── styles.js
│   ├── package.json
│   ├── vite.config.js
│   └── vercel.json
├── data/
│   ├── raw/                            # Original downloaded files
│   ├── clean/                          # Cleaned train/val/test CSVs
│   └── transformed/                    # Feature-engineered splits (with & without SMOTE)
├── tests/
│   ├── unit/
│   └── integration/
├── docs/                               # Project documentation PDFs
├── .env.example
├── Makefile
├── pyproject.toml
└── pytest.ini
```

---

## Quick Start

### Prerequisites

- Python ≥ 3.11
- [Poetry](https://python-poetry.org/docs/#installation)
- Node.js ≥ 18 (for the frontend)
- A [FRED API key](https://fred.stlouisfed.org/docs/api/api_key.html) (free)

### 1. Clone the repository

```bash
git clone https://github.com/abdulrahmannmostafa/Lend-Or-Debt.git
cd Lend-Or-Debt
```

### 2. Install Python dependencies

```bash
poetry install
```

### 3. Activate the virtual environment

```bash
poetry shell
```

### 4. Configure environment variables

```bash
cp .env.example .env
# Edit .env and fill in your FRED_API_KEY
```

### 5. Run the full pipeline

```bash
make pipeline
```

### 6. (Optional) Start the backend and frontend

```bash
# Backend
cd backend
pip install -r requirements.txt
python app.py

# Frontend (separate terminal)
cd frontend
npm install
npm run dev
```

---

## Pipeline Stages

Run all phases at once or execute them individually:

| Command | Description |
|---|---|
| `make pipeline` | Run all phases end-to-end |
| `make acquisition` | Phase 1 — download & merge raw data |
| `make validation` | Phase 2 — validate data with Great Expectations |
| `make cleaning` | Phase 3 — clean data and split into train/val/test |
| `make transformation` | Phase 4 — encode, scale, and apply SMOTE |
| `make eda` | Phase 5 — generate EDA plots and dashboards |

You can also resume the pipeline from any phase:

```bash
poetry run python -m src.pipeline.master_pipeline --from 3
```

---

## Model Training & Selection

Phase 6 trains a chosen model, logs metrics to MLflow, and supports feature selection strategies.

```bash
make modeling_selection model=<id> smote=<0|1> fs=<method> ver=<n> k=<n>
```

| Parameter | Description | Default |
|---|---|---|
| `model` | Model ID (see table below) | `6` |
| `smote` | Apply SMOTE oversampling (`1` = yes) | `0` |
| `fs` | Feature selection: `spearman`, `mutual_info`, `intersect` | `spearman` |
| `ver` | Run version number (for MLflow labelling) | `4` |
| `k` | Number of top features to select | `20` |

**Model IDs:**

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
| 8 | K-Nearest Neighbors |
| 9 | SVM |

**Example — train XGBoost with SMOTE and Spearman feature selection:**

```bash
make modeling_selection model=2 smote=1 fs=spearman ver=1 k=20
```

MLflow experiments are tracked under the `data_science_Project_team1` experiment. Start the UI with:

```bash
mlflow ui
```

---

## Backend API

The Flask backend (`backend/app.py`) exposes the following endpoints:

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/upload` | Upload a raw CSV dataset |
| `POST` | `/api/run/cleaning` | Run the data cleaning phase |
| `POST` | `/api/run/transformation` | Run the feature transformation phase |
| `POST` | `/api/eda/init` | Initialise the EDA engine |
| `GET` | `/api/eda/columns` | List available columns and feature groups |
| `POST` | `/api/eda/univariate` | Generate a univariate plot for a column |
| `POST` | `/api/eda/pie` | Generate a pie chart for a categorical feature |
| `POST` | `/api/eda/continuous` | Scatter / regression plot of two continuous features |
| `POST` | `/api/eda/discrete_vs_continuous` | Box / violin plot of discrete vs. continuous feature |
| `POST` | `/api/eda/discrete_vs_target` | Stacked bar chart of discrete feature vs. target |
| `POST` | `/api/eda/correlation` | Full correlation matrix heatmap |
| `POST` | `/api/eda/dashboard_with_smote` | EDA dashboard (post-SMOTE data) |
| `POST` | `/api/eda/dashboard_without_smote` | EDA dashboard (pre-SMOTE data) |
| `GET` | `/api/debug` | Runtime diagnostics |

All plot endpoints return a JSON response with a base64-encoded PNG image:

```json
{ "image": "<base64-string>" }
```

---

## Frontend Dashboard

The React dashboard (`frontend/`) lets you:

- **Upload** a CSV dataset
- **Run** cleaning and transformation phases via buttons
- **Explore** EDA plots interactively (univariate, pie, scatter, box, stacked bar, correlation matrix)
- **View** pre-rendered dashboards (with and without SMOTE)

Start in development mode:

```bash
cd frontend
npm install
npm run dev        # http://localhost:5173
npm run build      # production build
```

---

## Development Commands

```bash
# Linting
make lint          # Ruff code check
make format        # Ruff auto-format
make sta           # mypy static type analysis

# Testing
make unit          # Unit tests
make integration   # Integration tests

# Combined — lint, type-check, then test
poetry run ruff check src/ && poetry run mypy src/ && poetry run pytest tests/
```

---

## Environment Variables

| Variable | Description |
|---|---|
| `FRED_API_KEY` | API key for the Federal Reserve Economic Data (FRED) service |

Copy `.env.example` to `.env` and fill in the values before running the pipeline.

---

## Deployment

| Component | Platform | Config file |
|---|---|---|
| Backend (Flask) | [Railway](https://railway.app/) | `backend/Procfile`, `backend/railway.json` |
| Frontend (React) | [Vercel](https://vercel.com/) | `frontend/vercel.json` |
