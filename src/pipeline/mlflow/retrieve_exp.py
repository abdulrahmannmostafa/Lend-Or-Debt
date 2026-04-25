import mlflow
from loguru import logger

EXPERIMENT_NAME = r"data_science_Project_team1"

mlflow.set_tracking_uri("http://127.0.0.1:5000")  # Add this


if __name__ == "__main__":
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

    # Type Guard: Check if experiment exists
    if experiment is not None:
        logger.info("EXPERIMENT INFO:")
        logger.info(f"Name: {experiment.name}")
        logger.info(f"ID: {experiment.experiment_id}")
        logger.info(f"Artifact Location: {experiment.artifact_location}")
        logger.info(f"Tags: {experiment.tags}")
        logger.info(f"Lifecycle Stage: {experiment.lifecycle_stage}")
        logger.info(f"Creation timestamp: {experiment.creation_time}")
    else:
        logger.error(f"Error: Experiment with name '{EXPERIMENT_NAME}' not found.")
