.PHONY: scrape merge validate all help

help:
	@echo "Available commands:"
	@echo "  make acquisition - Run all data acquisition scripts in order"
	@echo "  make validation  - Run the validation pipeline"
	@echo "  make pipeline    - Run the full pipeline of all phases"

lint:
	@echo ">>> Running code linting..."
	poetry run poetry run ruff check src/
	@echo ">>> Code linting is completed"

format:
	@echo ">>> Running code formatting..."
	poetry run ruff format src/
	@echo ">>> Code formatting is completed"

sta:
	@echo ">>> Running static type analysis..."
	poetry run mypy src/
	@echo ">>> Static type analysis is completed"

unit:
	@echo ">>> Running unit tests..."
	poetry run pytest tests/unit/
	@echo ">>> Unit tests are completed"

integration:
	@echo ">>> Running integration tests..."
	poetry run pytest tests/integration/
	@echo ">>> Integration tests are completed"

acquisition:
	@echo ">>> Running data acquisition script..."
	poetry run python -m src.pipeline.data_acquisition
	@echo ">>> Data acquisition is completed"

validation:
	@echo ">>> Running data validation script..."
	poetry run python -m src.pipeline.data_validation
	@echo ">>> Data validation is completed"

cleaning:
	@echo ">>> Running data cleaning script..."
	poetry run python -m src.pipeline.data_cleaning
	@echo ">>> Data cleaning is completed"

transformation:
	@echo ">>> Running data transformation script..."
	poetry run python -m src.pipeline.data_transformation
	@echo ">>> Data transformation is completed"

pipeline:
	@echo ">>> Full pipeline of all phases"
	poetry run python -m src.pipeline.master_pipeline
	@echo ">>> All phases are completed"