.PHONY: scrape merge validate all help

help:
	@echo "Available commands:"
	@echo "  make acquisition - Run all data acquisition scripts in order"
	@echo "  make validation  - Run the validation pipeline"
	@echo "  make pipeline    - Run the full pipeline of all phases"

acquisition:
	@echo ">>> Running data acquisition script..."
	poetry run python -m src.pipeline.data_acquisition
	@echo ">>> Data acquisition is completed"

validation:
	@echo ">>> Running validation pipeline..."
	poetry run python -m src.pipeline.data_validation
	@echo ">>> Data validation is completed"

pipeline:
	@echo ">>> Full pipeline of all phases"
	poetry run python -m src.pipeline.master_pipeline
	@echo ">>> All phases are completed"