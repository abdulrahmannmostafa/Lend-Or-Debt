import mlflow
# from globals import EXPERIMENT_NAME

mlflow.set_tracking_uri("http://127.0.0.1:5000")
EXPERIMENT_NAME = r"data_science_Project_team1"

if __name__ == "__main__":
    experiment_id = mlflow.create_experiment(
        name=EXPERIMENT_NAME,
        artifact_location="my_mlflow_artifacts",
        tags={"env": "dev", "version": "1.0.0"},
    )

    print(experiment_id)
