import warnings
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from loguru import logger

from src.pipeline.config import (
    test_transformed_path_without_smote,
    train_transformed_path_without_smote,
    val_transformed_path_without_smote,
)
from src.pipeline.feature_extraction import FeatureExtractor
from src.pipeline.mlflow.models import (
    AdaBoostModel,
    LogisticRegressionModel,
    KNNModel,
    RFModel,
    SVMModel,
    XGBModel,
    LGBMModel,
    ExtraTreesModel,
    CatBoostModel,
    DecisionTreeModel,
)

plt.style.use("dark_background")
warnings.filterwarnings("ignore", category=UserWarning)


class DataLoader:
    def __init__(self):
        self.feature_extractor = FeatureExtractor()

    def load_data(self):
        self.train_data = pd.read_csv(train_transformed_path_without_smote)
        self.validation_data = pd.read_csv(val_transformed_path_without_smote)
        self.test_data = pd.read_csv(test_transformed_path_without_smote)

        target = "default payment next month"

        self.X_train, self.y_train = (
            self.train_data.drop(columns=[target]),
            self.train_data[target],
        )
        self.X_val, self.y_val = (
            self.validation_data.drop(columns=[target]),
            self.validation_data[target],
        )
        self.X_test, self.y_test = (
            self.test_data.drop(columns=[target]),
            self.test_data[target],
        )

        logger.success("Data Loaded Successfully")

    def prepare_features(self):
        self.X_train = self.feature_extractor.smart_scaler(self.X_train)
        self.X_val = self.feature_extractor.smart_scaler(self.X_val, 0)
        self.X_test = self.feature_extractor.smart_scaler(self.X_test, 0)

    def run_logistic(self):
        model = LogisticRegressionModel(
            self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test
        )
        model.model_train()
        model.model_predict()

    def run_rf(self):
        model = RFModel(
            self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test
        )
        model.model_train()
        model.model_predict()

    def run_xgb(self):
        model = XGBModel(
            self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test
        )
        model.model_train()
        model.model_predict()

    def run_lgbm(self):
        model = LGBMModel(
            self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test
        )
        model.model_train()
        model.model_predict()

    def run_et(self):
        model = ExtraTreesModel(
            self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test
        )
        model.model_train()
        model.model_predict()

    def run_catboost(self):
        model = CatBoostModel(
            self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test
        )
        model.model_train()
        model.model_predict()

    def run_dt(self):
        model = DecisionTreeModel(
            self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test
        )
        model.model_train()
        model.model_predict()

    def run_adaboost(self):
        model = AdaBoostModel(
            self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test
        )
        model.model_train()
        model.model_predict()

    def run_knn(self):
        model = KNNModel(
            self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test
        )
        model.model_train()
        model.model_predict()

    def run_svm(self):
        model = SVMModel(
            self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test
        )
        model.model_train()
        model.model_predict()

    def run_experiment(
        self, model_type, smote=True, feature_selection="spearman", version=1, k=20
    ):
        logger.info("Loading dataset")
        self.load_data()
        logger.info("Preparing features")
        self.prepare_features()

        results = self.feature_extractor.test_feature_selection_methods(
            self.X_train, self.y_train, k=k
        )
        if feature_selection == "intersect":
            selected = list(
                set(results["spearman"]["cols"]) & set(results["mutual_info"]["cols"])
            )
        else:
            selected = list(set(results[feature_selection]["cols"]))
        self.X_train = self.X_train[selected]
        self.X_val = self.X_val[selected]
        self.X_test = self.X_test[selected]

        model_map = {
            0: ("logistic", self.run_logistic),
            1: ("rf", self.run_rf),
            2: ("xgb", self.run_xgb),
            3: ("lgbm", self.run_lgbm),
            4: ("et", self.run_et),
            5: ("catboost", self.run_catboost),
            6: ("dt", self.run_dt),
            7: ("adaboost", self.run_adaboost),
            8: ("knn", self.run_knn),
            9: ("svm", self.run_svm),
        }

        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        mlflow.set_experiment("data_science_Project_team1")

        model_name, model_fn = model_map[model_type]

        run_name = f"{model_name}_v{version}_{feature_selection}_{'smote' if smote else 'no_smote'}"

        with mlflow.start_run(run_name=run_name):
            logger.info(f"Running: {run_name}")
            model_fn()
            mlflow.set_tag("model_type", model_name)
            mlflow.set_tag("smote_applied", smote)
            mlflow.set_tag("feature_selection", feature_selection)
            mlflow.set_tag("n_features_selected", len(selected))
            mlflow.log_param("version", version)


def main():
    

    dl = DataLoader()
    dl.run_experiment(
        model_type=6, smote=False, feature_selection="spearman", version=4, k=20
        )
    dl.run_experiment(
        model_type=6, smote=False, feature_selection="mutual_info", version=5, k=20
        )
    dl.run_experiment(
        model_type=6, smote=False, feature_selection="intersect", version=6, k=25
        )
    
    dl.run_experiment(
        model_type=7, smote=False, feature_selection="spearman", version=4, k=20
        )
    dl.run_experiment(
        model_type=7, smote=False, feature_selection="mutual_info", version=5, k=20
        )
    dl.run_experiment(
        model_type=7, smote=False, feature_selection="intersect", version=6, k=25
        )
    
    dl.run_experiment(
        model_type=8, smote=False, feature_selection="spearman", version=4, k=20
        )
    dl.run_experiment(
        model_type=8, smote=False, feature_selection="mutual_info", version=5, k=20
        )
    dl.run_experiment(
        model_type=8, smote=False, feature_selection="intersect", version=6, k=25
        )
    
    dl.run_experiment(
    model_type=9, smote=False, feature_selection="spearman", version=4, k=20
        )
    dl.run_experiment(
        model_type=9, smote=False, feature_selection="mutual_info", version=5, k=20
        )
    dl.run_experiment(
        model_type=9, smote=False, feature_selection="intersect", version=6, k=25
        )
    




if __name__ == "__main__":
    main()
