# 1. Standard libraries
import warnings
# sys.path.insert(
#     0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
# )

# 3. Third-party libraries
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from loguru import logger

# 4. Local imports
from src.pipeline.config import (
    test_transformed_path,
    train_transformed_path,
    val_transformed_path,
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

# 5. Global Configuration (Executable code goes AFTER all imports)
plt.style.use("dark_background")
warnings.filterwarnings("ignore", category=UserWarning)


class DataLoader:
    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self.logistic_regression_model = None
        self.knn_model = None
        self.svm_model = None
        self.rf_model = None
        self.xgboost = None
        self.naive_bayes_model = None

    def load_data(self):
        # Load datasets
        self.train_data = pd.read_csv(train_transformed_path)
        self.validation_data = pd.read_csv(val_transformed_path)
        self.test_data = pd.read_csv(test_transformed_path)

        target = "default payment next month"

        # Split features and labels
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

        # Log shapes using info
        logger.info("Data Loaded Successfully:")
        logger.info(f"  - Train shape: {self.X_train.shape}")
        logger.info(f"  - Val shape:   {self.X_val.shape}")
        logger.info(f"  - Test shape:  {self.X_test.shape}")

    def print_class_distribution(self):

        sets = [
            ("Train", self.y_train),
            ("Validation", self.y_val),
            ("Test", self.y_test),
        ]

        for name, series in sets:
            dist = series.value_counts(normalize=True) * 100
            count = len(series)
            logger.info(f"--- {name} Class Distribution (Total: {count}) ---")
            for label, val in dist.items():
                print(f" Class {label}: {val:.2f}%")

    def prepare_features(self):
        self.X_train_current = self.feature_extractor.smart_scaler(self.X_train)
        self.y_train_current = self.y_train
        self.X_val = self.feature_extractor.smart_scaler(self.X_val, 0)
        self.X_test = self.feature_extractor.smart_scaler(self.X_test, 0)
        # self.feature_extractor.spearman_correlation(self.X_train_current,self.y_train_current)

    def run_logistic(self):
        self.logistic_regression_model = LogisticRegressionModel(
            self.X_train_current,
            self.y_train_current,
            self.X_val,
            self.y_val,
            self.X_test,
            self.y_test,
        )
        self.logistic_regression_model.model_train()
        self.logistic_regression_model.model_predict()
        self.logistic_regression_model.plot_learning_curve_for_smote()

    # def run_poly_logistic(self):
    #     self.logistic_Pregression_model=LogisticRegressionPolynomialModel(self.X_train_current,self.y_train_current,self.X_val,self.y_val,self.X_test,self.y_test)
    #     self.logistic_Pregression_model.model_train()
    #     self.logistic_Pregression_model.model_predict()
    #     self.logistic_Pregression_model.plot_learning_curve()
    def run_knn(self):
        knn_model = KNNModel(
            self.X_train_current,
            self.y_train_current,
            self.X_val,
            self.y_val,
            self.X_test,
            self.y_test,
        )
        knn_model.model_train()
        knn_model.model_predict()  # 60
        knn_model.plot_learning_curve_for_smote()

    def run_xgb(self):
        xgb_model = XGBModel(
            self.X_train_current,
            self.y_train_current,
            self.X_val,
            self.y_val,
            self.X_test,
            self.y_test,
        )
        xgb_model.model_train()
        xgb_model.model_predict()
        xgb_model.plot_learning_curve_for_smote()

    def run_rf(self):
        rf_model = RFModel(
            self.X_train_current,
            self.y_train_current,
            self.X_val,
            self.y_val,
            self.X_test,
            self.y_test,
        )
        rf_model.model_train()
        rf_model.model_predict()
        rf_model.plot_learning_curve_for_smote()

    def run_lightboost(self):
        lgb_model = LGBMModel(
            self.X_train_current,
            self.y_train_current,
            self.X_val,
            self.y_val,
            self.X_test,
            self.y_test,
        )
        lgb_model.model_train()
        lgb_model.model_predict()
        lgb_model.plot_learning_curve_for_smote()

    def run_svm(self):
        svm_model = SVMModel(
            self.X_train_current,
            self.y_train_current,
            self.X_val,
            self.y_val,
            self.X_test,
            self.y_test,
        )
        svm_model.model_train()
        svm_model.model_predict()
        svm_model.plot_learning_curve_for_smote()

    def run_extratrees(self):
        extratrees_model = ExtraTreesModel(
            self.X_train_current,
            self.y_train_current,
            self.X_val,
            self.y_val,
            self.X_test,
            self.y_test,
        )
        extratrees_model.model_train()
        extratrees_model.model_predict()
        extratrees_model.plot_learning_curve_for_smote()

    def run_catboost(self):
        catboost_model = CatBoostModel(
            self.X_train_current,
            self.y_train_current,
            self.X_val,
            self.y_val,
            self.X_test,
            self.y_test,
        )
        catboost_model.model_train()
        catboost_model.model_predict()
        catboost_model.plot_learning_curve_for_smote()

    def run_adaboost(self):
        adaboost_model = AdaBoostModel(
            self.X_train_current,
            self.y_train_current,
            self.X_val,
            self.y_val,
            self.X_test,
            self.y_test,
        )
        adaboost_model.model_train()
        adaboost_model.model_predict()
        adaboost_model.plot_learning_curve_for_smote()

    def run_dt(self):
        decisiontree_model = DecisionTreeModel(
            self.X_train_current,
            self.y_train_current,
            self.X_val,
            self.y_val,
            self.X_test,
            self.y_test,
        )
        decisiontree_model.model_train()
        decisiontree_model.model_predict()
        decisiontree_model.plot_learning_curve_for_smote()

    def run_experiment(self, model_type):
        self.load_data()
        self.print_class_distribution()
        self.prepare_features()

        results = self.feature_extractor.test_feature_selection_methods(
            self.X_train_current, self.y_train_current, k=30
        )
        mi = set(results["mutual_info"]["cols"])
        sp = set(results["spearman"]["cols"])
        selected = list(mi & sp)

        self.X_train_current = self.X_train_current[selected]
        self.X_val = self.X_val[selected]
        self.X_test = self.X_test[selected]

        model_names = {
            0: "logistic",
            1: "knn",
            2: "rf",
            3: "xgb",
            4: "lgbm",
            5: "svm",
            6: "extratrees",
            7: "catboost",
            8: "adaboost",
            9: "decisiontrees",
        }
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        mlflow.set_experiment("data_science_Project_team1")

        with mlflow.start_run(run_name=model_names[model_type]):
            mlflow.log_param("model_type", model_names[model_type])
            mlflow.log_param("n_features_selected", len(selected))
            mlflow.log_param("feature_selection", "mutual_info & spearman")
            # mlflow.log_param("smote_applied", False)

            ################################################3
            if model_type == 0:
                self.run_logistic()
            elif model_type == 1:
                self.run_knn()
            elif model_type == 2:
                self.run_rf()
            elif model_type == 3:
                self.run_xgb()
            elif model_type == 4:
                self.run_lightboost()
            elif model_type == 5:
                self.run_svm()
            elif model_type == 6:
                self.run_extratrees()
            elif model_type == 7:
                self.run_catboost()
            elif model_type == 8:
                self.run_adaboost()
            elif model_type == 9:
                self.run_dt()


def main():
    dataloader = DataLoader()
    dataloader.run_experiment(model_type=5)


if __name__ == "__main__":
    main()
