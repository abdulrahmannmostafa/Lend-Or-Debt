# Standard libraries
import os

# Third-party libraries
from imblearn.pipeline import Pipeline
from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mlflow
import optuna
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    RandomForestClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
)
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.metrics import roc_curve, auc
# Local imports

# 1. Global Configuration
plt.style.use("dark_background")


class BaseModel:
    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.model = None
        self.best_params = None

    def train_final_model(self):
        # after hyperparameter tunning like in ml course
        X = pd.concat([self.X_train, self.X_val])
        y = pd.concat([self.y_train, self.y_val])
        self.model.fit(X, y)

    def find_best_threshold_mcc(self, proba, y_true=None):
        # I rely on mcc like lecture for imbalanced cases
        if (
            y_true is None
        ):  # used in predict function as second step after choosing the model
            y_true = self.y_test
        best_mcc, best_t = 0, 0.5
        for t in [0.3, 0.4, 0.5, 0.6]:
            y_pred = (proba > t).astype(int)
            mcc = matthews_corrcoef(y_true, y_pred)
            if mcc > best_mcc:
                best_mcc, best_t = mcc, t
        return best_mcc, best_t

    def cross_validate(self, scoring="f1_macro", cv=5):
        # this a step for registering overfitting beside the plot
        scores = cross_val_score(
            self.model, self.X_train, self.y_train, cv=cv, scoring=scoring
        )
        mlflow.log_metric("cv_f1_mean", scores.mean())
        mlflow.log_metric("cv_f1_std", scores.std())
        return scores

    def evaluate(self, proba):
        # This for predict in models to track in mlflow some metrices
        best_mcc, best_t = self.find_best_threshold_mcc(proba)
        y_pred = (proba > best_t).astype(
            int
        )  # convert from probability to final labels
        cm = confusion_matrix(self.y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        report = classification_report(self.y_test, y_pred, output_dict=True)
        # MLflow metrics
        mlflow.log_metric("mcc", best_mcc)
        mlflow.log_metric("accuracy", accuracy_score(self.y_test, y_pred))
        mlflow.log_metric("precision_macro", report["macro avg"]["precision"])
        mlflow.log_metric("recall_macro", report["macro avg"]["recall"])
        mlflow.log_metric("f1_macro", f1_score(self.y_test, y_pred, average="macro"))
        # Business metric
        mlflow.log_metric(
            "business_cost", fn * 10 + fp * 1
        )  # here we assume ratio 10/1
        return y_pred, cm, best_mcc, best_t

    def plot_roc(self, proba, model_name="model"):
        fpr, tpr, _ = roc_curve(self.y_test, proba)
        roc_auc = auc(fpr, tpr)
        mlflow.log_metric("roc_auc", roc_auc)
        path = f"/tmp/roc_curve_{model_name}.png"
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], "--")
        plt.title(f"ROC Curve - {model_name}")
        plt.legend()
        plt.savefig(path)
        plt.close()
        mlflow.log_artifact(path, artifact_path="plots")
        os.remove(path)

    def plot_confusion_matrix(self, cm, model_name="model"):
        path = f"/tmp/confusion_matrix_{model_name}.png"
        plt.figure()
        ConfusionMatrixDisplay(cm).plot()
        plt.title(f"Confusion Matrix - {model_name}")
        plt.savefig(path)
        plt.close()
        mlflow.log_artifact(path, artifact_path="plots")
        os.remove(path)

    def plot_learning_curve(self, model_name="model"):
        path = f"/tmp/learning_curve_{model_name}.png"
        model = (
            self.model.__class__(**self.best_params)
            if self.best_params
            else self.model.__class__()
        )
        train_sizes = []
        train_scores = []
        val_scores = []
        for p in np.linspace(0.1, 1.0, 10):
            n = int(p * len(self.X_train))
            train_sizes.append(n)
            X_sub = self.X_train[:n]
            y_sub = self.y_train[:n]
            model.fit(X_sub, y_sub)
            train_scores.append(f1_score(y_sub, model.predict(X_sub), average="macro"))
            val_scores.append(
                f1_score(self.y_val, model.predict(self.X_val), average="macro")
            )

        # with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        #     path = path
        plt.figure()
        plt.plot(train_sizes, train_scores, label="Train")
        plt.plot(train_sizes, val_scores, label="Validation")
        plt.title("Learning Curve")
        plt.legend()
        plt.savefig(path)
        plt.close()
        mlflow.log_artifact(path, artifact_path="plots")
        os.remove(path)

    def log_classification_report(self, y_pred, model_name="model"):
        report_text = classification_report(self.y_test, y_pred)
        path = f"/tmp/classification_report_{model_name}.txt"
        with open(path, "w") as f:
            f.write(report_text)
        mlflow.log_artifact(path, artifact_path="reports")
        os.remove(path)

    def log_error_analysis(self, y_pred, model_name="model"):
        y_pred_series = pd.Series(y_pred, index=self.y_test.index)
        mask = y_pred_series != self.y_test
        errors = self.X_test[mask].copy()
        errors["true"] = self.y_test[mask]
        errors["pred"] = y_pred_series[mask]
        path = f"/tmp/error_analysis_{model_name}.csv"
        errors.to_csv(path, index=False)
        mlflow.log_artifact(path, artifact_path="errors")
        os.remove(path)

    def log_and_register_model(self, model_name):
        mlflow.sklearn.log_model(self.model, "model")
        run_id = mlflow.active_run().info.run_id
        mlflow.register_model(f"runs:/{run_id}/model", model_name)


class LogisticRegressionModel(BaseModel):
    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test):
        super().__init__(X_train, y_train, X_val, y_val, X_test, y_test)
        self.best_logreg_params = None

    def objective_logreg(self, trial):
        solver = trial.suggest_categorical(
            "solver", ["liblinear", "saga", "lbfgs"]
        )  # algorithm to find best weight
        l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)  # penalty type
        params = {
            "C": trial.suggest_float(
                "C", 1e-3, 10.0, log=True
            ),  # inverse of regularization term
            "solver": solver,
            "max_iter": 10000,
            "random_state": 42,
            "class_weight": "balanced",
        }
        if solver == "lbfgs":
            l1_ratio = 0  # L2
        if solver == "liblinear":
            if l1_ratio not in [0, 1]:
                raise optuna.exceptions.TrialPruned()
        if solver == "saga":
            params["l1_ratio"] = l1_ratio
        model = LogisticRegression(**params)
        model.fit(self.X_train, self.y_train)
        proba = model.predict_proba(self.X_val)[:, 1]
        best_mcc, _ = self.find_best_threshold_mcc(proba, y_true=self.y_val)
        return best_mcc

    def model_train(self):
        logger.info("Training Logistic Regression")
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective_logreg, n_trials=50, n_jobs=2)
        self.best_logreg_params = study.best_params.copy()
        if self.best_logreg_params.get("solver") in ["lbfgs", "liblinear"]:
            self.best_logreg_params.pop("l1_ratio", None)
        self.model = LogisticRegression(
            **self.best_logreg_params, max_iter=10000, random_state=42
        )
        self.best_params = self.best_logreg_params
        self.train_final_model()
        mlflow.log_params(self.best_logreg_params)
        mlflow.log_param("model_name", "logistic_regression")
        self.cross_validate()
        logger.success("Training done")

    def model_predict(self):
        logger.info("Predicting Logistic Regression")
        proba = self.model.predict_proba(self.X_test)[:, 1]
        y_pred, cm, best_mcc, best_t = self.evaluate(proba)
        self.log_classification_report(y_pred, model_name="logistic")
        self.log_error_analysis(y_pred, model_name="logistic")
        self.plot_confusion_matrix(cm, model_name="logistic")
        self.plot_roc(proba, model_name="logistic")
        self.plot_learning_curve(model_name="logistic")
        self.log_and_register_model("logistic_regression")
        logger.success(f"Done | MCC={best_mcc:.4f} | T={best_t}")


########################################################################
class KNNModel(BaseModel):
    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test):
        super().__init__(X_train, y_train, X_val, y_val, X_test, y_test)
        self.best_params = None

    def objective_knn(self, trial):
        params = {
            "n_neighbors": trial.suggest_int("n_neighbors", 5, 30),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
            "metric": trial.suggest_categorical("metric", ["euclidean", "manhattan"]),
            "algorithm": trial.suggest_categorical(
                "algorithm", ["ball_tree", "kd_tree"]
            ),
        }

        model = KNeighborsClassifier(**params)
        model.fit(self.X_train, self.y_train)

        proba = model.predict_proba(self.X_val)[:, 1]

        best_mcc, _ = self.find_best_threshold_mcc(proba, y_true=self.y_val)

        return best_mcc

    def model_train(self):
        logger.info("Training KNN")

        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective_knn, n_trials=40, n_jobs=2)

        self.best_params = study.best_params.copy()

        self.model = KNeighborsClassifier(**self.best_params)

        self.train_final_model()

        # MLflow logging
        mlflow.log_params(self.best_params)
        mlflow.log_param("model_name", "knn")

        self.cross_validate()

        logger.success("Training done")

    def model_predict(self):
        logger.info("Predicting KNN")

        proba = self.model.predict_proba(self.X_test)[:, 1]

        y_pred, cm, best_mcc, best_t = self.evaluate(proba)

        self.log_classification_report(y_pred, model_name="knn")
        self.log_error_analysis(y_pred, model_name="knn")
        self.plot_confusion_matrix(cm, model_name="knn")
        self.plot_roc(proba, model_name="knn")
        self.plot_learning_curve(model_name="knn")

        self.log_and_register_model("knn")

        logger.success(f"Done | MCC={best_mcc:.4f} | T={best_t}")


class RFModel(BaseModel):
    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test):
        super().__init__(X_train, y_train, X_val, y_val, X_test, y_test)
        self.best_rf_params = None

    def objective_rf(self, trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_samples_split": trial.suggest_int("min_samples_split", 15, 50),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 15, 50),
        }
        model = RandomForestClassifier(
            **params, random_state=42, class_weight="balanced"
        )
        model.fit(self.X_train, self.y_train)
        proba = model.predict_proba(self.X_val)[:, 1]
        best_mcc, _ = self.find_best_threshold_mcc(proba, y_true=self.y_val)
        return best_mcc

    def model_train(self):
        logger.info("Training Random Forest")
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective_rf, n_trials=30, n_jobs=2)

        self.best_rf_params = study.best_params.copy()
        self.model = RandomForestClassifier(
            **self.best_rf_params, random_state=42, class_weight="balanced"
        )
        self.best_params = self.best_rf_params
        self.train_final_model()

        mlflow.log_params(self.best_rf_params)
        mlflow.log_param("model_name", "random_forest")
        self.cross_validate()
        logger.success("Training done")

    def model_predict(self):
        logger.info("Predicting Random Forest")
        proba = self.model.predict_proba(self.X_test)[:, 1]
        y_pred, cm, best_mcc, best_t = self.evaluate(proba)

        self.log_classification_report(y_pred, model_name="rf")
        self.log_error_analysis(y_pred, model_name="rf")
        self.plot_confusion_matrix(cm, model_name="rf")
        self.plot_roc(proba, model_name="rf")
        self.plot_learning_curve(model_name="rf")
        self.log_and_register_model("random_forest")
        logger.success(f"Done | MCC={best_mcc:.4f} | T={best_t}")


class SVMModel(BaseModel):
    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test):
        super().__init__(X_train, y_train, X_val, y_val, X_test, y_test)
        self.best_params = None
        self.feature_map = None

    def clean_data(self, X):
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        X = np.clip(X, -5, 5)
        return X

    def transform(self, X, fit=False):
        X = self.clean_data(X)
        if fit:
            return self.feature_map.fit_transform(X)
        return self.feature_map.transform(X)

    def objective_svm(self, trial):
        params = {
            "n_components": trial.suggest_int("n_components", 50, 120),
            "gamma": trial.suggest_float("gamma", 1e-3, 0.05, log=True),
            "C": trial.suggest_float("C", 0.05, 3.0, log=True),
        }

        feature_map = Nystroem(
            kernel="rbf",
            gamma=params["gamma"],
            n_components=params["n_components"],
            random_state=42,
        )

        X_tr = feature_map.fit_transform(self.clean_data(self.X_train))
        X_va = feature_map.transform(self.clean_data(self.X_val))

        model = LinearSVC(
            C=params["C"],
            class_weight="balanced",
            dual=False,
            max_iter=2000,
            random_state=42,
        )

        model.fit(X_tr, self.y_train)

        scores = model.decision_function(X_va)
        proba = (scores - scores.min()) / (scores.max() - scores.min() + 1e-7)

        best_mcc, _ = self.find_best_threshold_mcc(proba, self.y_val)

        return best_mcc

    def model_train(self):
        logger.info("Training SVM")

        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective_svm, n_trials=60, n_jobs=1)

        self.best_params = study.best_params.copy()

        self.feature_map = Nystroem(
            kernel="rbf",
            gamma=self.best_params["gamma"],
            n_components=self.best_params["n_components"],
            random_state=42,
        )

        self.model = LinearSVC(
            C=self.best_params["C"],
            class_weight="balanced",
            dual=False,
            max_iter=3000,
            random_state=42,
        )

        # combine data
        X_comb = pd.concat([self.X_train, self.X_val])
        y_comb = pd.concat([self.y_train, self.y_val])

        X_comb_tr = self.transform(X_comb, fit=True)
        self.model.fit(X_comb_tr, y_comb)

        mlflow.log_params(self.best_params)
        mlflow.log_param("model_name", "svm")

        logger.success("Training done")

    def plot_learning_curve(self, model_name="svm"):
        pipe = Pipeline(
            [
                (
                    "nystroem",
                    Nystroem(
                        kernel="rbf",
                        gamma=self.best_params["gamma"],
                        n_components=self.best_params["n_components"],
                        random_state=42,
                    ),
                ),
                (
                    "svc",
                    LinearSVC(
                        C=self.best_params["C"],
                        class_weight="balanced",
                        dual=False,
                        max_iter=3000,
                        random_state=42,
                    ),
                ),
            ]
        )

        X_train_clean = self.clean_data(self.X_train)

        train_sizes, train_scores, val_scores = learning_curve(
            pipe,
            X_train_clean,
            self.y_train,
            cv=5,
            scoring="f1_macro",
            train_sizes=np.linspace(0.1, 1.0, 8),
            n_jobs=2,
        )

        plt.figure(figsize=(8, 5))
        plt.plot(train_sizes, train_scores.mean(axis=1), "o-", label="Train F1")
        plt.plot(train_sizes, val_scores.mean(axis=1), "o-", label="Validation F1")
        plt.title(f"Learning Curve - {model_name}")
        plt.legend()
        plt.grid()

        filename = f"learning_curve_{model_name}.png"
        plt.savefig(filename)
        mlflow.log_artifact(filename)
        plt.close()

    def model_predict(self):
        logger.info("Predicting SVM")

        X_te = self.transform(self.X_test)

        scores = self.model.decision_function(X_te)
        proba = (scores - scores.min()) / (scores.max() - scores.min() + 1e-7)

        y_pred, cm, best_mcc, best_t = self.evaluate(proba)

        self.log_classification_report(y_pred, model_name="svm")
        self.log_error_analysis(y_pred, model_name="svm")
        self.plot_confusion_matrix(cm, model_name="svm")
        self.plot_roc(proba, model_name="svm")
        self.plot_learning_curve(model_name="svm")

        self.log_and_register_model("svm")

        logger.success(f"Done | MCC={best_mcc:.4f} | T={best_t}")


class XGBModel(BaseModel):
    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test):
        super().__init__(X_train, y_train, X_val, y_val, X_test, y_test)
        self.best_xgb_params = None
        self.scale_pos_weight = (self.y_train == 0).sum() / (self.y_train == 1).sum()

    def objective_xgb(self, trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 0.5, 5),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 1),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 10),
            "scale_pos_weight": self.scale_pos_weight,
            "eval_metric": "logloss",
            "random_state": 42,
        }
        model = XGBClassifier(**params)
        model.fit(self.X_train, self.y_train)
        proba = model.predict_proba(self.X_val)[:, 1]
        best_mcc, _ = self.find_best_threshold_mcc(proba, y_true=self.y_val)
        return best_mcc

    def model_train(self):
        logger.info("Training XGBoost")
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective_xgb, n_trials=200, n_jobs=2)

        self.best_xgb_params = study.best_params.copy()
        self.model = XGBClassifier(
            **self.best_xgb_params,
            scale_pos_weight=self.scale_pos_weight,
            eval_metric="logloss",
            random_state=42,
        )
        self.best_params = self.best_xgb_params
        self.train_final_model()

        mlflow.log_params(self.best_xgb_params)
        mlflow.log_param("model_name", "xgboost")
        mlflow.log_param("scale_pos_weight", self.scale_pos_weight)
        self.cross_validate()
        logger.success("Training done")

    def model_predict(self):
        logger.info("Predicting XGBoost")
        proba = self.model.predict_proba(self.X_test)[:, 1]
        y_pred, cm, best_mcc, best_t = self.evaluate(proba)

        self.log_classification_report(y_pred, model_name="xgb")
        self.log_error_analysis(y_pred, model_name="xgb")
        self.plot_confusion_matrix(cm, model_name="xgb")
        self.plot_roc(proba, model_name="xgb")
        self.plot_learning_curve(model_name="xgb")
        self.log_and_register_model("xgboost")
        logger.success(f"Done | MCC={best_mcc:.4f} | T={best_t}")


class ExtraTreesModel(BaseModel):
    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test):
        super().__init__(X_train, y_train, X_val, y_val, X_test, y_test)

    def objective_et(self, trial):
        param = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 400),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_samples_split": trial.suggest_int("min_samples_split", 20, 150),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 20, 150),
        }
        model = ExtraTreesClassifier(**param, random_state=42, class_weight="balanced")
        model.fit(self.X_train, self.y_train)
        proba = model.predict_proba(self.X_val)[:, 1]
        best_mcc, _ = self.find_best_threshold_mcc(proba, y_true=self.y_val)
        return best_mcc

    def model_train(self):
        logger.info("Training ExtraTrees")
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective_et, n_trials=130, n_jobs=2)

        self.best_params = study.best_params.copy()
        self.model = ExtraTreesClassifier(
            **self.best_params, random_state=42, class_weight="balanced"
        )
        self.train_final_model()

        mlflow.log_params(self.best_params)
        mlflow.log_param("model_name", "extra_trees")
        self.cross_validate()
        logger.success("Training done")

    def model_predict(self):
        logger.info("Predicting ExtraTrees")
        proba = self.model.predict_proba(self.X_test)[:, 1]
        y_pred, cm, best_mcc, best_t = self.evaluate(proba)

        self.log_classification_report(y_pred, model_name="et")
        self.log_error_analysis(y_pred, model_name="et")
        self.plot_confusion_matrix(cm, model_name="et")
        self.plot_roc(proba, model_name="et")
        self.plot_learning_curve(model_name="et")
        self.log_and_register_model("extra_trees")
        logger.success(f"Done | MCC={best_mcc:.4f} | T={best_t}")


class CatBoostModel(BaseModel):
    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test):
        super().__init__(X_train, y_train, X_val, y_val, X_test, y_test)
        self.best_params = None

    def objective_cat(self, trial):
        params = {
            "iterations": trial.suggest_int("iterations", 200, 600),
            "depth": trial.suggest_int("depth", 4, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 3, 20),
            "auto_class_weights": "Balanced",
            "verbose": 0,
            "random_seed": 42,
        }

        model = CatBoostClassifier(**params)
        model.fit(self.X_train, self.y_train)

        proba = model.predict_proba(self.X_val)[:, 1]

        best_mcc, _ = self.find_best_threshold_mcc(proba, y_true=self.y_val)

        return best_mcc

    def model_train(self):
        logger.info("Training CatBoost")

        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective_cat, n_trials=100, n_jobs=2)

        self.best_params = study.best_params.copy()

        self.model = CatBoostClassifier(
            **self.best_params,
            auto_class_weights="Balanced",
            verbose=0,
            random_seed=42,
        )

        self.train_final_model()

        # MLflow logging
        mlflow.log_params(self.best_params)
        mlflow.log_param("model_name", "catboost")

        self.cross_validate()

        logger.success("Training done")

    def model_predict(self):
        logger.info("Predicting CatBoost")
        proba = self.model.predict_proba(self.X_test)[:, 1]
        y_pred, cm, best_mcc, best_t = self.evaluate(proba)
        self.log_classification_report(y_pred, model_name="catboost")
        self.log_error_analysis(y_pred, model_name="catboost")
        self.plot_confusion_matrix(cm, model_name="catboost")
        self.plot_roc(proba, model_name="catboost")
        self.plot_learning_curve(model_name="catboost")
        self.log_and_register_model("catboost")
        logger.success(f"Done | MCC={best_mcc:.4f} | T={best_t}")


class LGBMModel(BaseModel):
    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test):
        super().__init__(X_train, y_train, X_val, y_val, X_test, y_test)
        self.scale_pos_weight = (self.y_train == 0).sum() / (self.y_train == 1).sum()
        self.scale_pos_weight = 1

    def objective_lgbm(self, trial):
        param = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 120),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3),
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "scale_pos_weight": self.scale_pos_weight,
            "verbosity": -1,
            "random_state": 42,
        }
        model = LGBMClassifier(**param)
        model.fit(self.X_train, self.y_train)
        proba = model.predict_proba(self.X_val)[:, 1]
        best_mcc, _ = self.find_best_threshold_mcc(proba, y_true=self.y_val)
        return best_mcc

    def model_train(self):
        logger.info("Training LightGBM")
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective_lgbm, n_trials=150, n_jobs=2)

        self.best_params = study.best_params.copy()
        self.model = LGBMClassifier(
            **self.best_params,
            scale_pos_weight=self.scale_pos_weight,
            verbosity=-1,
            random_state=42,
        )
        self.train_final_model()

        mlflow.log_params(self.best_params)
        mlflow.log_param("model_name", "lightgbm")
        mlflow.log_param("scale_pos_weight", self.scale_pos_weight)
        self.cross_validate()
        logger.success("Training done")

    def plot_feature_importance(self, model_name="lgbm", top_n=15):
        importances = self.model.feature_importances_
        features = self.X_train.columns

        fi_df = (
            pd.DataFrame({"feature": features, "importance": importances})
            .sort_values("importance", ascending=False)
            .head(top_n)
        )

        path = f"/tmp/feature_importance_{model_name}.png"
        plt.figure(figsize=(10, 6))
        plt.barh(fi_df["feature"][::-1], fi_df["importance"][::-1])
        plt.title(f"Top {top_n} Feature Importances - {model_name}")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        mlflow.log_artifact(path, artifact_path="plots")
        os.remove(path)

    def model_predict(self):
        logger.info("Predicting LightGBM")
        proba = self.model.predict_proba(self.X_test)[:, 1]
        y_pred, cm, best_mcc, best_t = self.evaluate(proba)

        self.log_classification_report(y_pred, model_name="lgbm")
        self.log_error_analysis(y_pred, model_name="lgbm")
        self.plot_confusion_matrix(cm, model_name="lgbm")
        self.plot_roc(proba, model_name="lgbm")
        self.plot_learning_curve(model_name="lgbm")

        self.plot_feature_importance(model_name="lgbm")
        self.log_and_register_model("lightgbm")
        logger.success(f"Done | MCC={best_mcc:.4f} | T={best_t}")


class AdaBoostModel(BaseModel):
    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test):
        super().__init__(X_train, y_train, X_val, y_val, X_test, y_test)
        self.best_params = None

    def objective_ada(self, trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 150),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
            "random_state": 42,
        }

        model = AdaBoostClassifier(**params)
        model.fit(self.X_train, self.y_train)

        proba = model.predict_proba(self.X_val)[:, 1]

        best_mcc, _ = self.find_best_threshold_mcc(proba, y_true=self.y_val)

        return best_mcc

    def model_train(self):
        logger.info("Training AdaBoost")

        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective_ada, n_trials=80, n_jobs=2)

        self.best_params = study.best_params.copy()

        self.model = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1, class_weight="balanced")
        )

        self.train_final_model()

        # MLflow logging
        mlflow.log_params(self.best_params)
        mlflow.log_param("model_name", "adaboost")

        self.cross_validate()

        logger.success("Training done")

    def model_predict(self):
        logger.info("Predicting AdaBoost")

        proba = self.model.predict_proba(self.X_test)[:, 1]

        y_pred, cm, best_mcc, best_t = self.evaluate(proba)

        self.log_classification_report(y_pred, model_name="adaboost")
        self.log_error_analysis(y_pred, model_name="adaboost")
        self.plot_confusion_matrix(cm, model_name="adaboost")
        self.plot_roc(proba, model_name="adaboost")
        self.plot_learning_curve(model_name="adaboost")

        self.log_and_register_model("adaboost")

        logger.success(f"Done | MCC={best_mcc:.4f} | T={best_t}")


class DecisionTreeModel(BaseModel):
    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test):
        super().__init__(X_train, y_train, X_val, y_val, X_test, y_test)
        self.best_params = None

    def objective_dt(self, trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "min_samples_split": trial.suggest_int("min_samples_split", 20, 100),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 20, 100),
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
            "class_weight": "balanced",
            "random_state": 42,
        }

        model = DecisionTreeClassifier(**params)
        model.fit(self.X_train, self.y_train)

        proba = model.predict_proba(self.X_val)[:, 1]

        best_mcc, _ = self.find_best_threshold_mcc(proba, y_true=self.y_val)

        return best_mcc

    def model_train(self):
        logger.info("Training DecisionTree")
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective_dt, n_trials=160, n_jobs=2)

        self.best_params = study.best_params.copy()

        self.model = DecisionTreeClassifier(
            **self.best_params,
            class_weight="balanced",
            random_state=42,
        )

        self.train_final_model()

        # MLflow logging
        mlflow.log_params(self.best_params)
        mlflow.log_param("model_name", "decision_tree")

        self.cross_validate()

        logger.success("Training done")

    def model_predict(self):
        logger.info("Predicting DecisionTree")

        proba = self.model.predict_proba(self.X_test)[:, 1]

        y_pred, cm, best_mcc, best_t = self.evaluate(proba)

        self.log_classification_report(y_pred, model_name="decision_tree")
        self.log_error_analysis(y_pred, model_name="decision_tree")
        self.plot_confusion_matrix(cm, model_name="decision_tree")
        self.plot_roc(proba, model_name="decision_tree")
        self.plot_learning_curve(model_name="decision_tree")

        self.log_and_register_model("decision_tree")

        logger.success(f"Done | MCC={best_mcc:.4f} | T={best_t}")
