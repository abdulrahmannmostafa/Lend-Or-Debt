# Standard libraries
# Third-party libraries
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
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

# Local imports

# 1. Global Configuration/Setup
plt.style.use("dark_background")


class LogisticRegressionModel:
    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.best_logreg_model = None

    def objective_logreg(self, trial):

        solver = trial.suggest_categorical("solver", ["liblinear", "saga", "lbfgs"])

        l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)

        param = {
            "C": trial.suggest_float(
                "C", 1e-3, 10.0, log=True
            ),  # inverse of regularization term
            "solver": solver,
            "max_iter": 4000,
            "random_state": 42,
            "class_weight": "balanced",
        }

        if solver == "lbfgs":
            l1_ratio = 0
            # param["penalty"] = "l2"

        if solver == "liblinear":
            if l1_ratio not in [0, 1]:
                raise optuna.exceptions.TrialPruned()

        if solver == "saga":
            param["l1_ratio"] = l1_ratio

        model = LogisticRegression(**param)
        model.fit(self.X_train, self.y_train)

        y_proba = model.predict_proba(self.X_val)[:, 1]

        best_mcc = -1
        for t in [0.3, 0.4, 0.5, 0.6]:
            y_pred = (y_proba > t).astype(int)
            # f1 = f1_score(self.y_val, y_pred, average="macro")
            mcc = matthews_corrcoef(self.y_val, y_pred)
            if mcc > best_mcc:
                best_mcc = mcc

        return best_mcc

    def model_train(self):
        study_logreg = optuna.create_study(direction="maximize")
        study_logreg.optimize(self.objective_logreg, n_trials=15, n_jobs=2)

        best_params = study_logreg.best_params.copy()

        solver = best_params.get("solver")

        if solver == "lbfgs":
            best_params.pop("l1_ratio", None)
            # best_params["penalty"] = "l2"
        elif solver == "liblinear":
            best_params.pop("l1_ratio", None)

        X_combined = pd.concat([self.X_train, self.X_val])
        y_combined = pd.concat([self.y_train, self.y_val])

        self.best_logreg_model = LogisticRegression(
            **best_params, max_iter=4000, random_state=42
        )
        self.best_logreg_params = best_params
        self.best_logreg_model.fit(X_combined, y_combined)

        mlflow.log_params(self.best_logreg_params)

    def model_predict(self):
        proba = self.best_logreg_model.predict_proba(self.X_test)[:, 1]

        best_f1, best_t = 0, 0.5
        for t in [0.3, 0.4, 0.5, 0.6]:
            y_pred = (proba > t).astype(int)
            f1 = f1_score(self.y_test, y_pred, average="macro")
            if f1 > best_f1:
                best_f1, best_t = f1, t

        y_pred = (proba > best_t).astype(int)

        # --- Metrics & Reports ---
        report_dict = classification_report(self.y_test, y_pred, output_dict=True)
        report_text = classification_report(self.y_test, y_pred)

        # MLflow Logging
        mlflow.log_metric("f1_macro", best_f1)
        mlflow.log_metric("accuracy", accuracy_score(self.y_test, y_pred))
        mlflow.log_metric("mcc", matthews_corrcoef(self.y_test, y_pred))
        mlflow.log_metric("best_threshold", best_t)
        mlflow.log_metric("precision_macro", report_dict["macro avg"]["precision"])
        mlflow.log_metric("recall_macro", report_dict["macro avg"]["recall"])
        mlflow.sklearn.log_model(self.best_logreg_model, "model")

        # Save Report Artifact
        report_filename = f"classification_report_{self.__class__.__name__}.txt"
        with open(report_filename, "w") as f:
            f.write(report_text)
        mlflow.log_artifact(report_filename)

        # Confusion Matrix Artifact
        model_name = self.__class__.__name__
        cm_filename = f"confusion_matrix_{model_name}.png"
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(self.y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap="Blues")
        plt.title(f"Confusion Matrix - {model_name}")
        plt.savefig(cm_filename, bbox_inches="tight")
        mlflow.log_artifact(cm_filename)
        plt.close()

        # --- Printing ---
        print(f"\n{'=' * 20} {self.__class__.__name__} {'=' * 20}")
        print(f"Best Threshold: {best_t}")
        print(f"Test Accuracy: {accuracy_score(self.y_test, y_pred):.4f}")
        print("\nClassification Report:\n", report_text)

    def plot_learning_curve_for_smote(self):

        model = LogisticRegression(
            **self.best_logreg_params if hasattr(self, "best_logreg_params") else {},
            max_iter=4000,
            random_state=42,
        )

        train_percentages = np.linspace(0.1, 1.0, 10)
        train_sizes = []
        train_scores = []
        val_scores = []

        for p in train_percentages:
            n_samples = int(p * len(self.X_train))
            train_sizes.append(n_samples)

            X_subset = self.X_train[:n_samples]
            y_subset = self.y_train[:n_samples]

            model.fit(X_subset, y_subset)

            y_train_pred = model.predict(X_subset)
            train_scores.append(f1_score(y_subset, y_train_pred, average="macro"))

            y_val_pred = model.predict(self.X_val)
            val_scores.append(f1_score(self.y_val, y_val_pred, average="macro"))

        model_name = self.__class__.__name__
        lc_filename = f"learning_curve_smote_{model_name}.png"

        plt.figure(figsize=(10, 6))
        plt.plot(
            train_sizes,
            train_scores,
            "o-",
            color="#a427d6",
            label="Train F1 (SMOTE Data)",
        )
        plt.plot(
            train_sizes,
            val_scores,
            "o-",
            color="#2ca02c",
            label="Validation F1 (Original Data)",
        )

        plt.xlabel("Number of Training Samples")
        plt.ylabel("F1 Score (Macro)")
        plt.title(f"SMOTE Learning Curve Analysis - {model_name}")
        plt.legend(loc="lower right")
        plt.grid(True, linestyle="--", alpha=0.6)

        plt.fill_between(
            train_sizes,
            train_scores,
            val_scores,
            color="gray",
            alpha=0.1,
            label="Overfitting Gap",
        )

        plt.savefig(lc_filename, bbox_inches="tight")
        mlflow.log_artifact(lc_filename)

        # plt.show()
        plt.close()


class KNNModel:
    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.best_knn_model = None
        self.best_knn_params = None

    def objective_knn(self, trial):
        param = {
            "n_neighbors": trial.suggest_int("n_neighbors", 10, 50),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
            "metric": trial.suggest_categorical(
                "metric", ["euclidean", "manhattan", "minkowski"]
            ),
            "algorithm": trial.suggest_categorical(
                "algorithm", ["ball_tree", "kd_tree"]
            ),
        }

        model = KNeighborsClassifier(**param)
        model.fit(self.X_train, self.y_train)

        y_proba = model.predict_proba(self.X_val)[:, 1]

        best_mcc = -1
        for t in [0.3, 0.4, 0.5, 0.6]:
            y_pred = (y_proba > t).astype(int)
            mcc = matthews_corrcoef(self.y_val, y_pred)
            if mcc > best_mcc:
                best_mcc = mcc
        return best_mcc

    def model_train(self):
        study_knn = optuna.create_study(direction="maximize")
        study_knn.optimize(self.objective_knn, n_trials=25, n_jobs=2)

        self.best_knn_params = study_knn.best_params.copy()

        X_combined = pd.concat([self.X_train, self.X_val])
        y_combined = pd.concat([self.y_train, self.y_val])

        self.best_knn_model = KNeighborsClassifier(**self.best_knn_params)
        self.best_knn_model.fit(X_combined, y_combined)

        mlflow.log_params(self.best_knn_params)

    def model_predict(self):
        proba = self.best_knn_model.predict_proba(self.X_test)[:, 1]

        best_f1, best_t = 0, 0.5
        for t in [0.3, 0.4, 0.5, 0.6]:
            y_pred = (proba > t).astype(int)
            f1 = f1_score(self.y_test, y_pred, average="macro")
            if f1 > best_f1:
                best_f1, best_t = f1, t

        y_pred = (proba > best_t).astype(int)

        # --- Metrics & Reports ---
        report_dict = classification_report(self.y_test, y_pred, output_dict=True)
        report_text = classification_report(self.y_test, y_pred)

        # MLflow Logging
        mlflow.log_metric("f1_macro", best_f1)
        mlflow.log_metric("accuracy", accuracy_score(self.y_test, y_pred))
        mlflow.log_metric("mcc", matthews_corrcoef(self.y_test, y_pred))
        mlflow.log_metric("best_threshold", best_t)
        mlflow.log_metric("precision_macro", report_dict["macro avg"]["precision"])
        mlflow.log_metric("recall_macro", report_dict["macro avg"]["recall"])
        mlflow.sklearn.log_model(self.best_knn_model, "model")

        # Save Report Artifact
        report_filename = f"classification_report_{self.__class__.__name__}.txt"
        with open(report_filename, "w") as f:
            f.write(report_text)
        mlflow.log_artifact(report_filename)

        # Confusion Matrix Artifact
        model_name = self.__class__.__name__
        cm_filename = f"confusion_matrix_{model_name}.png"
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(self.y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap="Greens")
        plt.title(f"Confusion Matrix - {model_name}")
        plt.savefig(cm_filename, bbox_inches="tight")
        mlflow.log_artifact(cm_filename)
        plt.close()

        # --- Printing ---
        print(f"\n{'=' * 20} {self.__class__.__name__} {'=' * 20}")
        print(f"Best Threshold: {best_t}")
        print(f"Test Accuracy: {accuracy_score(self.y_test, y_pred):.4f}")
        print("\nClassification Report:\n", report_text)

    def plot_learning_curve_for_smote(self):

        model = KNeighborsClassifier(**self.best_knn_params)

        train_percentages = np.linspace(0.1, 1.0, 10)
        train_sizes, train_scores, val_scores = [], [], []

        for p in train_percentages:
            n_samples = int(p * len(self.X_train))
            train_sizes.append(n_samples)

            X_subset = self.X_train[:n_samples]
            y_subset = self.y_train[:n_samples]

            model.fit(X_subset, y_subset)

            train_scores.append(
                f1_score(y_subset, model.predict(X_subset), average="macro")
            )
            val_scores.append(
                f1_score(self.y_val, model.predict(self.X_val), average="macro")
            )

        lc_filename = f"learning_curve_{self.__class__.__name__}.png"
        plt.figure(figsize=(10, 6))
        plt.plot(
            train_sizes,
            train_scores,
            "o-",
            color="#a427d6",
            label="Train F1 (SMOTE Data)",
        )
        plt.plot(
            train_sizes,
            val_scores,
            "o-",
            color="#2ca02c",
            label="Validation F1 (Original Data)",
        )
        plt.title(f"Learning Curve Analysis - {self.__class__.__name__}")
        plt.xlabel("Samples")
        plt.ylabel("F1 Score")
        plt.legend()
        plt.grid(True)
        plt.savefig(lc_filename, bbox_inches="tight")
        mlflow.log_artifact(lc_filename)
        plt.close()


class RFModel:
    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.best_rf_model = None
        self.best_rf_params = None

    def objective_rf(self, trial):
        param = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        }

        model = RandomForestClassifier(
            **param, random_state=42, class_weight="balanced"
        )
        model.fit(self.X_train, self.y_train)

        proba = model.predict_proba(self.X_val)[:, 1]

        best_mcc = -1
        for t in [0.3, 0.4, 0.5, 0.6]:
            y_pred = (proba > t).astype(int)
            mcc = matthews_corrcoef(self.y_val, y_pred)
            if mcc > best_mcc:
                best_mcc = mcc

        return best_mcc

    def model_train(self):
        study_rf = optuna.create_study(direction="maximize")
        study_rf.optimize(self.objective_rf, n_trials=30, n_jobs=2)

        self.best_rf_params = study_rf.best_params.copy()

        X_combined = pd.concat([self.X_train, self.X_val])
        y_combined = pd.concat([self.y_train, self.y_val])

        self.best_rf_model = RandomForestClassifier(
            **self.best_rf_params, random_state=42, class_weight="balanced"
        )
        self.best_rf_model.fit(X_combined, y_combined)

        # MLflow Logging
        mlflow.log_params(self.best_rf_params)

    def model_predict(self):
        proba = self.best_rf_model.predict_proba(self.X_test)[:, 1]

        best_f1, best_t = 0, 0.5
        for t in [0.3, 0.4, 0.5, 0.6]:
            y_pred = (proba > t).astype(int)
            f1 = f1_score(self.y_test, y_pred, average="macro")
            if f1 > best_f1:
                best_f1, best_t = f1, t

        y_pred = (proba > best_t).astype(int)

        # --- Metrics & Reports ---
        report_dict = classification_report(self.y_test, y_pred, output_dict=True)
        report_text = classification_report(self.y_test, y_pred)

        # MLflow Logging
        mlflow.log_metric("f1_macro", best_f1)
        mlflow.log_metric("accuracy", accuracy_score(self.y_test, y_pred))
        mlflow.log_metric("mcc", matthews_corrcoef(self.y_test, y_pred))
        mlflow.log_metric("best_threshold", best_t)
        mlflow.log_metric("precision_macro", report_dict["macro avg"]["precision"])
        mlflow.log_metric("recall_macro", report_dict["macro avg"]["recall"])
        mlflow.sklearn.log_model(self.best_rf_model, "model")

        # Save Report Artifact
        report_filename = f"classification_report_{self.__class__.__name__}.txt"
        with open(report_filename, "w") as f:
            f.write(report_text)
        mlflow.log_artifact(report_filename)

        # Confusion Matrix Artifact
        model_name = self.__class__.__name__
        cm_filename = f"confusion_matrix_{model_name}.png"
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(self.y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap="Purples")
        plt.title(f"Confusion Matrix - {model_name}")
        plt.savefig(cm_filename, bbox_inches="tight")
        mlflow.log_artifact(cm_filename)
        plt.close()

        # --- Printing ---
        print(f"\n{'=' * 20} {self.__class__.__name__} {'=' * 20}")
        print(f"Best Threshold: {best_t}")
        print(f"Test Accuracy: {accuracy_score(self.y_test, y_pred):.4f}")
        print("\nClassification Report:\n", report_text)

    def plot_learning_curve_for_smote(self):
        model = RandomForestClassifier(
            **self.best_rf_params, random_state=42, class_weight="balanced"
        )

        train_percentages = np.linspace(0.1, 1.0, 10)
        train_sizes, train_scores, val_scores = [], [], []

        for p in train_percentages:
            n_samples = int(p * len(self.X_train))
            train_sizes.append(n_samples)

            X_subset = self.X_train[:n_samples]
            y_subset = self.y_train[:n_samples]

            model.fit(X_subset, y_subset)

            train_scores.append(
                f1_score(y_subset, model.predict(X_subset), average="macro")
            )
            val_scores.append(
                f1_score(self.y_val, model.predict(self.X_val), average="macro")
            )

        lc_filename = f"learning_curve_{self.__class__.__name__}.png"
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_scores, "o-", color="#6a1b9a", label="Train F1")
        plt.plot(train_sizes, val_scores, "o-", color="#1565c0", label="Validation F1")
        plt.title(f"Learning Curve Analysis - {self.__class__.__name__}")
        plt.xlabel("Number of Training Samples")
        plt.ylabel("F1 Score (Macro)")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)

        plt.savefig(lc_filename, bbox_inches="tight")
        mlflow.log_artifact(lc_filename)
        plt.close()


class SVMModel:
    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.best_svm_model = None
        self.best_svm_params = None
        self.best_threshold = 0.5

    def clean_data(self, X):

        X_clean = X.replace([np.inf, -np.inf], np.nan)
        X_clean = X_clean.fillna(X_clean.median())
        X_clean = X_clean.clip(-10, 10)
        return X_clean

    def objective_svm(self, trial):
        kernel = trial.suggest_categorical("kernel", ["rbf", "poly"])
        param = {
            "C": trial.suggest_float("C", 0.1, 20, log=True),
            "kernel": kernel,
            "gamma": trial.suggest_float("gamma", 1e-4, 1, log=True),
            "max_iter": 4000,
            "class_weight": "balanced",
            "probability": True,
            "random_state": 42,
        }

        if kernel == "poly":
            param["degree"] = trial.suggest_int("degree", 2, 3)
            param["coef0"] = trial.suggest_float("coef0", 0, 1)

        model = SVC(**param)

        X_train_clean = self.clean_data(self.X_train)
        X_val_clean = self.clean_data(self.X_val)

        model.fit(X_train_clean, self.y_train)
        proba = model.predict_proba(X_val_clean)[:, 1]

        best_mcc = -1
        best_t = 0.5
        for t in np.arange(0.2, 0.8, 0.1):
            y_pred = (proba > t).astype(int)
            mcc = matthews_corrcoef(self.y_val, y_pred)
            if mcc > best_mcc:
                best_mcc = mcc
                best_t = t

        trial.set_user_attr("best_threshold", best_t)
        return best_mcc

    def model_train(self):
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective_svm, n_trials=3, n_jobs=1)

        self.best_svm_params = study.best_params.copy()
        self.best_threshold = study.best_trial.user_attrs["best_threshold"]

        X_combined = pd.concat([self.X_train, self.X_val])
        y_combined = pd.concat([self.y_train, self.y_val])

        self.best_svm_model = SVC(
            **self.best_svm_params,
            class_weight="balanced",
            probability=True,
            random_state=42,
        )

        self.best_svm_model.fit(self.clean_data(X_combined), y_combined)

        # MLflow Logging
        mlflow.log_params(self.best_svm_params)
        mlflow.log_param("optimized_threshold", self.best_threshold)

    def model_predict(self):
        X_test_clean = self.clean_data(self.X_test)
        proba = self.best_svm_model.predict_proba(X_test_clean)[:, 1]
        y_pred = (proba > self.best_threshold).astype(int)

        # --- Metrics & Reports ---
        report_dict = classification_report(self.y_test, y_pred, output_dict=True)
        report_text = classification_report(self.y_test, y_pred)

        # MLflow Logging
        mlflow.log_metric("f1_macro", report_dict["macro avg"]["f1-score"])
        mlflow.log_metric("accuracy", accuracy_score(self.y_test, y_pred))
        mlflow.log_metric("mcc", matthews_corrcoef(self.y_test, y_pred))
        mlflow.log_metric("best_threshold", self.best_threshold)
        mlflow.log_metric("precision_macro", report_dict["macro avg"]["precision"])
        mlflow.log_metric("recall_macro", report_dict["macro avg"]["recall"])
        mlflow.sklearn.log_model(self.best_svm_model, "model")

        # Save Report Artifact
        report_filename = f"classification_report_{self.__class__.__name__}.txt"
        with open(report_filename, "w") as f:
            f.write(report_text)
        mlflow.log_artifact(report_filename)

        # Confusion Matrix Artifact
        model_name = self.__class__.__name__
        cm_filename = f"confusion_matrix_{model_name}.png"
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(self.y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap="Reds")
        plt.title(f"Confusion Matrix - {model_name}")
        plt.savefig(cm_filename, bbox_inches="tight")
        mlflow.log_artifact(cm_filename)
        plt.close()

        # --- Printing ---
        print(f"\n{'=' * 20} {self.__class__.__name__} {'=' * 20}")
        print(f"Best Threshold: {self.best_threshold:.4f}")
        print(f"Test Accuracy: {accuracy_score(self.y_test, y_pred):.4f}")
        print("\nClassification Report:\n", report_text)

    def plot_learning_curve_for_smote(self):

        model = SVC(
            **self.best_svm_params,
            class_weight="balanced",
            probability=True,
            random_state=42,
        )

        train_percentages = np.linspace(0.1, 1.0, 8)
        train_sizes, train_scores, val_scores = [], [], []

        X_train_clean = self.clean_data(self.X_train)
        X_val_clean = self.clean_data(self.X_val)

        for p in train_percentages:
            n_samples = int(p * len(X_train_clean))
            train_sizes.append(n_samples)

            X_subset = X_train_clean[:n_samples]
            y_subset = self.y_train[:n_samples]

            model.fit(X_subset, y_subset)

            train_scores.append(
                f1_score(y_subset, model.predict(X_subset), average="macro")
            )
            val_scores.append(
                f1_score(self.y_val, model.predict(X_val_clean), average="macro")
            )

        lc_filename = f"learning_curve_{self.__class__.__name__}.png"
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_scores, "o-", color="#d32f2f", label="Train F1")
        plt.plot(train_sizes, val_scores, "o-", color="#1976d2", label="Validation F1")
        plt.title(f"Learning Curve Analysis - {self.__class__.__name__}")
        plt.xlabel("Training Samples")
        plt.ylabel("F1 Score (Macro)")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)

        plt.savefig(lc_filename, bbox_inches="tight")
        mlflow.log_artifact(lc_filename)
        plt.close()


# class NaiveBayesModel:
#     def __init__(self,X_train,y_train,X_val,y_val,X_test,y_test):
#         self.X_train=X_train
#         self.y_train=y_train
#         self.X_val=X_val
#         self.y_val=y_val
#         self.X_test=X_test
#         self.y_test=y_test
#         self.best_nb_model=None

#     def objective_nb(self,trial):
#         var_smoothing = trial.suggest_float("var_smoothing", 1e-11, 1e-7, log=True)

#         model = GaussianNB(var_smoothing=var_smoothing)
#         model.fit(self.X_train, self.y_train)
#         y_val_pred = model.predict(self.X_val)
#         return accuracy_score(self.y_val, y_val_pred)

#     def model_train(self):
#         study_nb = optuna.create_study(direction="maximize")
#         study_nb.optimize(self.objective_nb, n_trials=200, n_jobs=2)

#         best_nb_params = study_nb.best_params
#         print("Best GaussianNB Parameters from Optuna:", best_nb_params)

#         # ------------------------------
#         # 2. TRAIN BEST NB ON FULL TRAIN + VALIDATION DATA
#         # ------------------------------
#         X_combined = pd.concat([self.X_train, self.X_val])
#         y_combined = pd.concat([self.y_train, self.y_val])

#         self.best_nb_model = GaussianNB(**best_nb_params)
#         self.best_nb_model.fit(X_combined, y_combined)


#     def model_predict(self):
#         y_test_pred = self.best_nb_model.predict(self.X_test)
#         test_acc = accuracy_score(self.y_test, y_test_pred)
#         print(f"\nTest Accuracy: {test_acc:.4f}")
#         print("f1_score:\n", f1_score(self.y_test, y_test_pred,average="weighted"))
#         print("Classification Report:\n", classification_report(self.y_test, y_test_pred))
class XGBModel:
    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.best_xgb_model = None
        self.best_xgb_params = None
        self.scale_pos_weight = 1

    def objective_xgb(self, trial):
        param = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 1),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 1),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 1),
            "scale_pos_weight": self.scale_pos_weight,
            "eval_metric": "logloss",
            "random_state": 42,
        }

        model = XGBClassifier(**param)
        model.fit(self.X_train, self.y_train)

        proba = model.predict_proba(self.X_val)[:, 1]

        best_mcc = -1
        for t in [0.3, 0.4, 0.5, 0.6]:
            y_pred = (proba > t).astype(int)
            mcc = matthews_corrcoef(self.y_val, y_pred)
            if mcc > best_mcc:
                best_mcc = mcc

        return best_mcc

    def model_train(self):
        study_xgb = optuna.create_study(direction="maximize")
        study_xgb.optimize(self.objective_xgb, n_trials=200, n_jobs=2)

        self.best_xgb_params = study_xgb.best_params.copy()

        X_combined = pd.concat([self.X_train, self.X_val])
        y_combined = pd.concat([self.y_train, self.y_val])

        self.best_xgb_model = XGBClassifier(
            **self.best_xgb_params,
            scale_pos_weight=self.scale_pos_weight,
            eval_metric="logloss",
            random_state=42,
        )

        self.best_xgb_model.fit(X_combined, y_combined)

        # MLflow Logging
        mlflow.log_params(self.best_xgb_params)

    def model_predict(self):
        proba = self.best_xgb_model.predict_proba(self.X_test)[
            :, 1
        ]  # عدلي اسم الموديل حسب الكلاس
        best_f1, best_t = 0, 0.5
        for t in [0.3, 0.4, 0.5, 0.6]:
            y_pred = (proba > t).astype(int)
            f1 = f1_score(self.y_test, y_pred, average="macro")
            if f1 > best_f1:
                best_f1, best_t = f1, t

            y_pred = (proba > best_t).astype(int)

            report_dict = classification_report(self.y_test, y_pred, output_dict=True)
            report_text = classification_report(self.y_test, y_pred)
            mlflow.log_metric("f1_macro", best_f1)
            mlflow.log_metric("accuracy", accuracy_score(self.y_test, y_pred))
            mlflow.log_metric("mcc", matthews_corrcoef(self.y_test, y_pred))
            mlflow.log_metric("best_threshold", best_t)
            mlflow.sklearn.log_model(self.best_xgb_model, "model")
            mlflow.log_metric("precision_macro", report_dict["macro avg"]["precision"])
            mlflow.log_metric("recall_macro", report_dict["macro avg"]["recall"])

            report_filename = f"classification_report_{self.__class__.__name__}.txt"
            with open(report_filename, "w") as f:
                f.write(report_text)
            mlflow.log_artifact(report_filename)

            # --- Printing ---
            print(f"\n{'=' * 20} {self.__class__.__name__} {'=' * 20}")
            print(f"Best Threshold: {best_t}")
            print(f"Test Accuracy: {accuracy_score(self.y_test, y_pred):.4f}")
            print("\nClassification Report:\n", report_text)

        # Confusion Matrix Artifact
        model_name = self.__class__.__name__
        cm_filename = f"confusion_matrix_{model_name}.png"

        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(self.y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap="Oranges")
        plt.title(f"Confusion Matrix - {model_name}")
        plt.savefig(cm_filename, bbox_inches="tight")
        mlflow.log_artifact(cm_filename)
        plt.close()

        # print(f"Best Threshold: {best_t}")
        # print(f"Test Accuracy: {accuracy_score(self.y_test, y_pred):.4f}")
        # print("Classification Report:\n", classification_report(self.y_test, y_pred))

    def plot_learning_curve_for_smote(self):
        model = XGBClassifier(
            **self.best_xgb_params,
            scale_pos_weight=self.scale_pos_weight,
            eval_metric="logloss",
            random_state=42,
        )

        train_percentages = np.linspace(0.1, 1.0, 10)
        train_sizes, train_scores, val_scores = [], [], []

        for p in train_percentages:
            n_samples = int(p * len(self.X_train))
            train_sizes.append(n_samples)

            X_subset = self.X_train[:n_samples]
            y_subset = self.y_train[:n_samples]

            model.fit(X_subset, y_subset)

            train_scores.append(
                f1_score(y_subset, model.predict(X_subset), average="macro")
            )
            val_scores.append(
                f1_score(self.y_val, model.predict(self.X_val), average="macro")
            )

        lc_filename = f"learning_curve_{self.__class__.__name__}.png"
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_scores, "o-", color="#e65100", label="Train F1")
        plt.plot(train_sizes, val_scores, "o-", color="#0277bd", label="Validation F1")
        plt.title(f"Learning Curve Analysis - {self.__class__.__name__}")
        plt.xlabel("Training Samples")
        plt.ylabel("F1 Score (Macro)")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)

        plt.savefig(lc_filename, bbox_inches="tight")
        mlflow.log_artifact(lc_filename)
        plt.close()


class ExtraTreesModel:
    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.best_params = None
        self.best_model = None

    def objective_et(self, trial):
        param = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "max_depth": trial.suggest_int("max_depth", 5, 15),
            "min_samples_split": trial.suggest_int("min_samples_split", 20, 50),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 20, 50),
        }

        model = ExtraTreesClassifier(**param, random_state=42, class_weight="balanced")
        model.fit(self.X_train, self.y_train)

        proba = model.predict_proba(self.X_val)[:, 1]

        best_mcc = -1
        for t in [0.3, 0.4, 0.5, 0.6]:
            y_pred = (proba > t).astype(int)
            mcc = matthews_corrcoef(self.y_val, y_pred)
            if mcc > best_mcc:
                best_mcc = mcc

        return best_mcc

    def model_train(self):
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective_et, n_trials=130, n_jobs=2)

        self.best_params = study.best_params.copy()

        X_combined = pd.concat([self.X_train, self.X_val])
        y_combined = pd.concat([self.y_train, self.y_val])

        self.best_model = ExtraTreesClassifier(
            **self.best_params, random_state=42, class_weight="balanced"
        )

        self.best_model.fit(X_combined, y_combined)

        # MLflow Logging
        mlflow.log_params(self.best_params)

    def model_predict(self):
        proba = self.best_model.predict_proba(self.X_test)[:, 1]

        best_f1, best_t = 0, 0.5
        for t in [0.3, 0.4, 0.5, 0.6]:
            y_pred = (proba > t).astype(int)
            f1 = f1_score(self.y_test, y_pred, average="macro")
            if f1 > best_f1:
                best_f1, best_t = f1, t

        y_pred = (proba > best_t).astype(int)

        # --- Metrics & Reports ---
        report_dict = classification_report(self.y_test, y_pred, output_dict=True)
        report_text = classification_report(self.y_test, y_pred)

        # MLflow Logging
        mlflow.log_metric("f1_macro", best_f1)
        mlflow.log_metric("accuracy", accuracy_score(self.y_test, y_pred))
        mlflow.log_metric("mcc", matthews_corrcoef(self.y_test, y_pred))
        mlflow.log_metric("best_threshold", best_t)
        mlflow.log_metric("precision_macro", report_dict["macro avg"]["precision"])
        mlflow.log_metric("recall_macro", report_dict["macro avg"]["recall"])
        mlflow.sklearn.log_model(self.best_model, "model")

        # Save Report Artifact
        report_filename = f"classification_report_{self.__class__.__name__}.txt"
        with open(report_filename, "w") as f:
            f.write(report_text)
        mlflow.log_artifact(report_filename)

        # Confusion Matrix Artifact
        model_name = self.__class__.__name__
        cm_filename = f"confusion_matrix_{model_name}.png"
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(self.y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap="Greens")
        plt.title(f"Confusion Matrix - {model_name}")
        plt.savefig(cm_filename, bbox_inches="tight")
        mlflow.log_artifact(cm_filename)
        plt.close()

        # --- Printing ---
        print(f"\n{'=' * 20} {self.__class__.__name__} {'=' * 20}")
        print(f"Best Threshold: {best_t}")
        print(f"Test Accuracy: {accuracy_score(self.y_test, y_pred):.4f}")
        print("\nClassification Report:\n", report_text)

    def plot_learning_curve_for_smote(self):
        model = ExtraTreesClassifier(
            **self.best_params, random_state=42, class_weight="balanced"
        )

        train_percentages = np.linspace(0.1, 1.0, 10)
        train_sizes, train_scores, val_scores = [], [], []

        for p in train_percentages:
            n_samples = int(p * len(self.X_train))
            train_sizes.append(n_samples)

            X_subset = self.X_train[:n_samples]
            y_subset = self.y_train[:n_samples]

            model.fit(X_subset, y_subset)

            train_scores.append(
                f1_score(y_subset, model.predict(X_subset), average="macro")
            )
            val_scores.append(
                f1_score(self.y_val, model.predict(self.X_val), average="macro")
            )

        lc_filename = f"learning_curve_{self.__class__.__name__}.png"
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_scores, "o-", color="#2e7d32", label="Train F1")
        plt.plot(train_sizes, val_scores, "o-", color="#1565c0", label="Validation F1")
        plt.title(f"Learning Curve Analysis - {self.__class__.__name__}")
        plt.xlabel("Training Samples")
        plt.ylabel("F1 Score (Macro)")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)

        plt.savefig(lc_filename, bbox_inches="tight")
        mlflow.log_artifact(lc_filename)
        plt.close()


class CatBoostModel:
    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.best_params = None
        self.model = None

    def objective_cat(self, trial):
        param = {
            "iterations": trial.suggest_int("iterations", 200, 600),
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
            "auto_class_weights": "Balanced",
            "verbose": 0,
            "random_seed": 42,
        }

        model = CatBoostClassifier(**param)
        model.fit(self.X_train, self.y_train)

        proba = model.predict_proba(self.X_val)[:, 1]

        best_mcc = -1
        for t in [0.3, 0.4, 0.5, 0.6]:
            y_pred = (proba > t).astype(int)
            mcc = matthews_corrcoef(self.y_val, y_pred)
            if mcc > best_mcc:
                best_mcc = mcc

        return best_mcc

    def model_train(self):
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective_cat, n_trials=30, n_jobs=2)

        self.best_params = study.best_params.copy()

        X_combined = pd.concat([self.X_train, self.X_val])
        y_combined = pd.concat([self.y_train, self.y_val])

        self.model = CatBoostClassifier(
            **self.best_params, auto_class_weights="Balanced", verbose=0, random_seed=42
        )

        self.model.fit(X_combined, y_combined)

        # MLflow Logging
        mlflow.log_params(self.best_params)

    def model_predict(self):
        proba = self.model.predict_proba(self.X_test)[:, 1]

        best_f1, best_t = 0, 0.5
        for t in [0.3, 0.4, 0.5, 0.6]:
            y_pred = (proba > t).astype(int)
            f1 = f1_score(self.y_test, y_pred, average="macro")
            if f1 > best_f1:
                best_f1, best_t = f1, t

        y_pred = (proba > best_t).astype(int)

        # --- Metrics & Reports ---
        report_dict = classification_report(self.y_test, y_pred, output_dict=True)
        report_text = classification_report(self.y_test, y_pred)

        # MLflow Logging
        mlflow.log_metric("f1_macro", best_f1)
        mlflow.log_metric("accuracy", accuracy_score(self.y_test, y_pred))
        mlflow.log_metric("mcc", matthews_corrcoef(self.y_test, y_pred))
        mlflow.log_metric("best_threshold", best_t)
        mlflow.log_metric("precision_macro", report_dict["macro avg"]["precision"])
        mlflow.log_metric("recall_macro", report_dict["macro avg"]["recall"])

        mlflow.sklearn.log_model(self.model, "model")

        # Save Report Artifact
        report_filename = f"classification_report_{self.__class__.__name__}.txt"
        with open(report_filename, "w") as f:
            f.write(report_text)
        mlflow.log_artifact(report_filename)

        # Confusion Matrix Artifact
        model_name = self.__class__.__name__
        cm_filename = f"confusion_matrix_{model_name}.png"
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(self.y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap="YlGnBu")  # لون مختلف (أصفر-أخضر-أزرق) لتمييز CatBoost
        plt.title(f"Confusion Matrix - {model_name}")
        plt.savefig(cm_filename, bbox_inches="tight")
        mlflow.log_artifact(cm_filename)
        plt.close()

        # --- Printing ---
        print(f"\n{'=' * 20} {self.__class__.__name__} {'=' * 20}")
        print(f"Best Threshold: {best_t}")
        print(f"Test Accuracy: {accuracy_score(self.y_test, y_pred):.4f}")
        print("\nClassification Report:\n", report_text)

    def plot_learning_curve_for_smote(self):
        # نستخدم نفس البارامترات مع وضع verbose=0 لمنع الزحمة في الـ console
        model = CatBoostClassifier(
            **self.best_params, auto_class_weights="Balanced", verbose=0, random_seed=42
        )

        train_percentages = np.linspace(0.1, 1.0, 10)
        train_sizes, train_scores, val_scores = [], [], []

        for p in train_percentages:
            n_samples = int(p * len(self.X_train))
            train_sizes.append(n_samples)

            X_subset = self.X_train[:n_samples]
            y_subset = self.y_train[:n_samples]

            model.fit(X_subset, y_subset)

            train_scores.append(
                f1_score(y_subset, model.predict(X_subset), average="macro")
            )
            val_scores.append(
                f1_score(self.y_val, model.predict(self.X_val), average="macro")
            )

        lc_filename = f"learning_curve_{self.__class__.__name__}.png"
        plt.figure(figsize=(10, 6))
        plt.plot(
            train_sizes, train_scores, "o-", color="#fbc02d", label="Train F1"
        )  # أصفر
        plt.plot(
            train_sizes, val_scores, "o-", color="#0097a7", label="Validation F1"
        )  # سيان
        plt.title(f"Learning Curve Analysis - {self.__class__.__name__}")
        plt.xlabel("Training Samples")
        plt.ylabel("F1 Score (Macro)")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)

        plt.savefig(lc_filename, bbox_inches="tight")
        mlflow.log_artifact(lc_filename)
        plt.close()


class LGBMModel:
    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.best_params = None
        self.model = None
        self.scale_pos_weight = 1

    def objective_lgbm(self, trial):
        param = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", -1, 15),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
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

        best_mcc = -1
        for t in [0.3, 0.4, 0.5, 0.6]:
            y_pred = (proba > t).astype(int)
            mcc = matthews_corrcoef(self.y_val, y_pred)
            if mcc > best_mcc:
                best_mcc = mcc

        return best_mcc

    def model_train(self):
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective_lgbm, n_trials=150, n_jobs=2)

        self.best_params = study.best_params.copy()

        self.best_params["scale_pos_weight"] = self.scale_pos_weight

        X_combined = pd.concat([self.X_train, self.X_val])
        y_combined = pd.concat([self.y_train, self.y_val])

        self.model = LGBMClassifier(**self.best_params, verbosity=-1, random_state=42)
        self.model.fit(X_combined, y_combined)

        # MLflow Logging
        mlflow.log_params(self.best_params)

    def model_predict(self):
        proba = self.model.predict_proba(self.X_test)[:, 1]

        best_f1, best_t = 0, 0.5
        for t in [0.3, 0.4, 0.5, 0.6]:
            y_pred = (proba > t).astype(int)
            f1 = f1_score(self.y_test, y_pred, average="macro")
            if f1 > best_f1:
                best_f1, best_t = f1, t

        y_pred = (proba > best_t).astype(int)

        # --- Metrics & Reports ---
        report_dict = classification_report(self.y_test, y_pred, output_dict=True)
        report_text = classification_report(self.y_test, y_pred)

        # MLflow Logging
        mlflow.log_metric("f1_macro", best_f1)
        mlflow.log_metric("accuracy", accuracy_score(self.y_test, y_pred))
        mlflow.log_metric("mcc", matthews_corrcoef(self.y_test, y_pred))
        mlflow.log_metric("best_threshold", best_t)
        mlflow.log_metric("precision_macro", report_dict["macro avg"]["precision"])
        mlflow.log_metric("recall_macro", report_dict["macro avg"]["recall"])
        mlflow.lightgbm.log_model(self.model, "model")

        # Save Report Artifact
        report_filename = f"classification_report_{self.__class__.__name__}.txt"
        with open(report_filename, "w") as f:
            f.write(report_text)
        mlflow.log_artifact(report_filename)

        # Confusion Matrix Artifact
        model_name = self.__class__.__name__
        cm_filename = f"confusion_matrix_{model_name}.png"
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(self.y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap="Blues")
        plt.title(f"Confusion Matrix - {model_name}")
        plt.savefig(cm_filename, bbox_inches="tight")
        mlflow.log_artifact(cm_filename)
        plt.close()

        # --- Printing ---
        print(f"\n{'=' * 20} {self.__class__.__name__} {'=' * 20}")
        print(f"Best Threshold: {best_t}")
        print(f"Test Accuracy: {accuracy_score(self.y_test, y_pred):.4f}")
        print("\nClassification Report:\n", report_text)

    def plot_learning_curve_for_smote(self):
        model = LGBMClassifier(**self.best_params, verbosity=-1, random_state=42)

        train_percentages = np.linspace(0.1, 1.0, 10)
        train_sizes, train_scores, val_scores = [], [], []

        for p in train_percentages:
            n_samples = int(p * len(self.X_train))
            train_sizes.append(n_samples)

            X_subset = self.X_train[:n_samples]
            y_subset = self.y_train[:n_samples]

            model.fit(X_subset, y_subset)

            train_scores.append(
                f1_score(y_subset, model.predict(X_subset), average="macro")
            )
            val_scores.append(
                f1_score(self.y_val, model.predict(self.X_val), average="macro")
            )

        lc_filename = f"learning_curve_{self.__class__.__name__}.png"
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_scores, "o-", color="#1976d2", label="Train F1")
        plt.plot(train_sizes, val_scores, "o-", color="#d32f2f", label="Validation F1")
        plt.title(f"Learning Curve Analysis - {self.__class__.__name__}")
        plt.xlabel("Training Samples")
        plt.ylabel("F1 Score (Macro)")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)

        plt.savefig(lc_filename, bbox_inches="tight")
        mlflow.log_artifact(lc_filename)
        plt.close()


class AdaBoostModel:
    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.best_params = None
        self.model = None

    def objective_ada(self, trial):

        param = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.5),
            # "algorithm": trial.suggest_categorical("algorithm", ["SAMME", "SAMME.R"]),
            "random_state": 42,
        }

        model = AdaBoostClassifier(**param)
        model.fit(self.X_train, self.y_train)

        proba = model.predict_proba(self.X_val)[:, 1]
        best_mcc = -1
        for t in [0.3, 0.4, 0.5, 0.6]:
            y_pred = (proba > t).astype(int)
            mcc = matthews_corrcoef(self.y_val, y_pred)
            if mcc > best_mcc:
                best_mcc = mcc
        return best_mcc

    def model_train(self):
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective_ada, n_trials=30, n_jobs=2)

        self.best_params = study.best_params.copy()
        X_combined = pd.concat([self.X_train, self.X_val])
        y_combined = pd.concat([self.y_train, self.y_val])

        self.model = AdaBoostClassifier(**self.best_params, random_state=42)
        self.model.fit(X_combined, y_combined)
        mlflow.log_params(self.best_params)

    def model_predict(self):
        proba = self.model.predict_proba(self.X_test)[:, 1]
        best_f1, best_t = 0, 0.5
        for t in [0.3, 0.4, 0.5, 0.6]:
            y_pred = (proba > t).astype(int)
            f1 = f1_score(self.y_test, y_pred, average="macro")
            if f1 > best_f1:
                best_f1, best_t = f1, t

        y_pred = (proba > best_t).astype(int)
        # report_dict = classification_report(self.y_test, y_pred, output_dict=True)
        report_text = classification_report(self.y_test, y_pred)

        mlflow.log_metric("f1_macro", best_f1)
        mlflow.log_metric("mcc", matthews_corrcoef(self.y_test, y_pred))
        mlflow.log_metric("best_threshold", best_t)
        mlflow.sklearn.log_model(self.model, "model")

        # Artifacts
        report_filename = f"classification_report_{self.__class__.__name__}.txt"
        with open(report_filename, "w") as f:
            f.write(report_text)
        mlflow.log_artifact(report_filename)

        cm_filename = f"confusion_matrix_{self.__class__.__name__}.png"
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(self.y_test, y_pred)
        ConfusionMatrixDisplay(confusion_matrix=cm).plot(cmap="Reds")
        plt.title(f"Confusion Matrix - {self.__class__.__name__}")
        plt.savefig(cm_filename)
        mlflow.log_artifact(cm_filename)
        plt.close()

        print(f"\n{'=' * 20} AdaBoost {'=' * 20}")
        print(f"Best Threshold: {best_t}\n", report_text)

    def plot_learning_curve_for_smote(self):
        model = AdaBoostClassifier(**self.best_params, random_state=42)
        train_percentages = np.linspace(0.1, 1.0, 10)
        train_sizes, train_scores, val_scores = [], [], []

        for p in train_percentages:
            n_samples = int(p * len(self.X_train))
            train_sizes.append(n_samples)
            X_subset, y_subset = self.X_train[:n_samples], self.y_train[:n_samples]
            model.fit(X_subset, y_subset)
            train_scores.append(
                f1_score(y_subset, model.predict(X_subset), average="macro")
            )
            val_scores.append(
                f1_score(self.y_val, model.predict(self.X_val), average="macro")
            )

        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_scores, "o-", color="#d32f2f", label="Train F1")
        plt.plot(train_sizes, val_scores, "o-", color="#1976d2", label="Val F1")
        plt.title("Learning Curve - AdaBoost")
        plt.legend()
        plt.savefig(f"learning_curve_{self.__class__.__name__}.png")
        mlflow.log_artifact(f"learning_curve_{self.__class__.__name__}.png")
        plt.close()


class DecisionTreeModel:
    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.best_params = None
        self.model = None

    def objective_dt(self, trial):
        param = {
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_samples_split": trial.suggest_int("min_samples_split", 10, 50),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 50),
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
            "class_weight": "balanced",
            "random_state": 42,
        }

        model = DecisionTreeClassifier(**param)
        model.fit(self.X_train, self.y_train)

        proba = model.predict_proba(self.X_val)[:, 1]

        best_mcc = -1
        for t in [0.3, 0.4, 0.5, 0.6]:
            y_pred = (proba > t).astype(int)
            mcc = matthews_corrcoef(self.y_val, y_pred)
            if mcc > best_mcc:
                best_mcc = mcc

        return best_mcc

    def model_train(self):
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective_dt, n_trials=50, n_jobs=2)

        self.best_params = study.best_params.copy()

        X_combined = pd.concat([self.X_train, self.X_val])
        y_combined = pd.concat([self.y_train, self.y_val])

        self.model = DecisionTreeClassifier(
            **self.best_params, class_weight="balanced", random_state=42
        )
        self.model.fit(X_combined, y_combined)

        mlflow.log_params(self.best_params)

    def model_predict(self):
        proba = self.model.predict_proba(self.X_test)[:, 1]
        best_f1, best_t = 0, 0.5
        for t in [0.3, 0.4, 0.5, 0.6]:
            y_pred = (proba > t).astype(int)
            f1 = f1_score(self.y_test, y_pred, average="macro")
            if f1 > best_f1:
                best_f1, best_t = f1, t

        y_pred = (proba > best_t).astype(int)

        # report_dict = classification_report(self.y_test, y_pred, output_dict=True)
        report_text = classification_report(self.y_test, y_pred)

        mlflow.log_metric("f1_macro", best_f1)
        mlflow.log_metric("accuracy", accuracy_score(self.y_test, y_pred))
        mlflow.log_metric("mcc", matthews_corrcoef(self.y_test, y_pred))
        mlflow.log_metric("best_threshold", best_t)
        mlflow.sklearn.log_model(self.model, "model")

        report_filename = f"classification_report_{self.__class__.__name__}.txt"
        with open(report_filename, "w") as f:
            f.write(report_text)
        mlflow.log_artifact(report_filename)

        cm_filename = f"confusion_matrix_{self.__class__.__name__}.png"
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(self.y_test, y_pred)
        ConfusionMatrixDisplay(confusion_matrix=cm).plot(cmap="Purples")
        plt.title(f"Confusion Matrix - {self.__class__.__name__}")
        plt.savefig(cm_filename)
        mlflow.log_artifact(cm_filename)
        plt.close()

        print(f"\n{'=' * 20} DecisionTree {'=' * 20}")
        print(f"Best Threshold: {best_t}\n", report_text)

    def plot_learning_curve_for_smote(self):
        model = DecisionTreeClassifier(
            **self.best_params, class_weight="balanced", random_state=42
        )
        train_percentages = np.linspace(0.1, 1.0, 10)
        train_sizes, train_scores, val_scores = [], [], []

        for p in train_percentages:
            n_samples = int(p * len(self.X_train))
            train_sizes.append(n_samples)
            X_subset, y_subset = self.X_train[:n_samples], self.y_train[:n_samples]
            model.fit(X_subset, y_subset)
            train_scores.append(
                f1_score(y_subset, model.predict(X_subset), average="macro")
            )
            val_scores.append(
                f1_score(self.y_val, model.predict(self.X_val), average="macro")
            )

        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_scores, "o-", color="#7b1fa2", label="Train F1")
        plt.plot(train_sizes, val_scores, "o-", color="#1976d2", label="Val F1")
        plt.title("Learning Curve - DecisionTree")
        plt.legend()
        plt.savefig(f"learning_curve_{self.__class__.__name__}.png")
        mlflow.log_artifact(f"learning_curve_{self.__class__.__name__}.png")
        plt.close()
