#  Data Science Project

_A production-style end-to-end data pipeline for acquisition, validation, and analysis._

---

## Overview

This project implements a **modular data science pipeline** covering:

- Data Acquisition
- Data Validation
- End-to-End Pipeline Execution

Built with:

- **Poetry** for dependency management
- **Modular pipeline architecture**
- **Makefile Configuration** for reproducible workflows

---

## Quick Start

### 1- Clone the Repository

```bash
git clone https://github.com/abdulrahmannmostafa/Data-Science-Project.git
cd Data-Science-Project
```

### 2️- Install Dependencies

```bash
poetry install
```

### 3️- Activate Virtual Environment

```bash
poetry shell
```

### 4️- Configure Environment Variables

```bash
cp .env.example .env
```

Then add your secrets

---

## Running the Pipeline

### Run Full Pipeline (Recommended)

```bash
make pipeline
```

This will execute:

- Data acquisition
- Data Validation
- Full pipeline

---

## Run Individual Stages

### Data Acquisition

```bash
make acquisition
```

### Data Validation

```bash
make validation
```

---

## Project Structure

```bash
src/
 └── pipeline/
      ├── data_acquisition.py
      ├── data_validation.py
      └── master_pipeline.py
```

- `data_acquisition.py` - Fetches and prepares raw data
- `data_validation.py` - Runs validation checks
- `master_pipeline.py` - Orchestrates the full workflow

---

## Tips

Run modules properly:

```bash
poetry run python -m pipeline.master_pipeline
```

---

## Example Workflow

```bash
# Full pipeline
make pipeline

# Debug a single stage
make data_validation
```

---
