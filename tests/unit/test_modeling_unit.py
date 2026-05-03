import pytest
import pandas as pd
import numpy as np
import sys
import os
from unittest.mock import patch # for mlflow mocking
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from src.pipeline.mlflow.models import BaseModel, LogisticRegressionModel
from src.pipeline.mlflow.model_selection_evaluations import DataLoader


@pytest.fixture
def data_loader():
    return DataLoader()

#proper MLflow disable (context manager)
@pytest.fixture(autouse=True)
def disable_mlflow(monkeypatch):
    class DummyRun:
        class info:
            run_id = "dummy_run_id"         # ← needed for active_run().info.run_id
        def __enter__(self): return self
        def __exit__(self, *args): pass

    monkeypatch.setattr("mlflow.set_tracking_uri", lambda *a, **k: None)
    monkeypatch.setattr("mlflow.set_experiment", lambda *a, **k: None)
    monkeypatch.setattr("mlflow.start_run", lambda *a, **k: DummyRun())
    monkeypatch.setattr("mlflow.end_run", lambda *a, **k: None)
    monkeypatch.setattr("mlflow.active_run", lambda *a, **k: DummyRun())  # ← for run_id
    monkeypatch.setattr("mlflow.log_param", lambda *a, **k: None)
    monkeypatch.setattr("mlflow.log_params", lambda *a, **k: None)
    monkeypatch.setattr("mlflow.log_metric", lambda *a, **k: None)
    monkeypatch.setattr("mlflow.log_metrics", lambda *a, **k: None)
    monkeypatch.setattr("mlflow.set_tag", lambda *a, **k: None)
    monkeypatch.setattr("mlflow.set_tags", lambda *a, **k: None)
    monkeypatch.setattr("mlflow.log_artifact", lambda *a, **k: None)      # ← for plots/reports
    monkeypatch.setattr("mlflow.log_artifacts", lambda *a, **k: None)
    monkeypatch.setattr("mlflow.sklearn.log_model", lambda *a, **k: None) # ← model saving
    monkeypatch.setattr("mlflow.register_model", lambda *a, **k: None)    # ← model registry
# Data loading
def test_load_data(data_loader):
    data_loader.load_data()
    assert data_loader.X_train is not None
    assert data_loader.y_train is not None
    assert len(data_loader.X_train) == len(data_loader.y_train)

# Feature preparation
def test_prepare_features_all_sets(data_loader):
    data_loader.load_data()
    data_loader.prepare_features()
    for df in [data_loader.X_train, data_loader.X_val, data_loader.X_test]:
        assert not df.isnull().any().any()
        assert df.shape[0] > 0

#
def test_scaling_changes_data(data_loader):
    data_loader.load_data()
    before = data_loader.X_train.copy()
    data_loader.prepare_features()
    assert not before.equals(data_loader.X_train)


def test_row_count_preserved_after_prepare(data_loader):
    data_loader.load_data()
    n = len(data_loader.X_train)
    data_loader.prepare_features()
    assert len(data_loader.X_train) == n

# Feature selection

def test_feature_selection_simple(data_loader):
    data_loader.load_data()
    data_loader.prepare_features()
    results = data_loader.feature_extractor.test_feature_selection_methods(
        data_loader.X_train, data_loader.y_train, k=5
    )
    assert "spearman" in results
    assert len(results["spearman"]["cols"]) == 5


def test_feature_selection_applied_shape(data_loader):
    data_loader.load_data()
    data_loader.prepare_features()
    results = data_loader.feature_extractor.test_feature_selection_methods(
        data_loader.X_train, data_loader.y_train, k=5
    )

    selected = results["spearman"]["cols"]
    X_new = data_loader.X_train[selected]

    assert X_new.shape[1] == 5

# Mlflow Pipeline execution

@patch.object(DataLoader, "run_dt")
def test_pipeline_flow(mock_run):
    dl = DataLoader()
    dl.run_experiment(model_type=6, smote=False,
                      feature_selection="spearman", version=1, k=5)
    assert mock_run.called


@patch.object(DataLoader, "run_dt")
def test_selected_features_used(mock_run):
    dl = DataLoader()
    dl.run_experiment(model_type=6, smote=False,
                      feature_selection="spearman", version=1, k=5)
    assert dl.X_train.shape[1] == 5


@patch.object(DataLoader, "run_dt")
def test_feature_consistency_between_sets_features(mock_run):
    dl = DataLoader()
    dl.run_experiment(model_type=6, smote=False,
                      feature_selection="spearman", version=1, k=5)
    assert list(dl.X_train.columns) == list(dl.X_val.columns)
    assert list(dl.X_train.columns) == list(dl.X_test.columns)


@patch.object(DataLoader, "run_dt")
def test_intersect_feature_selection(mock_run):
    dl = DataLoader()
    dl.run_experiment(model_type=6, smote=False,
                      feature_selection="intersect", version=1, k=5)
    assert dl.X_train.shape[1] <= 5
    
def test_invalid_model_type():
    dl = DataLoader()
    with pytest.raises(KeyError):
        dl.run_experiment(
            model_type=99,
            smote=False,
            feature_selection="spearman",
            version=1,
            k=5
        )


# ================= BaseModel =================

def test_find_best_threshold():
    X = pd.DataFrame(np.random.rand(20, 3))
    y = pd.Series(np.random.randint(0, 2, 20))
    model = BaseModel(X, y, X, y, X, y)

    proba = np.random.rand(20)
    mcc, t = model.find_best_threshold_mcc(proba, y)

    assert 0 <= t <= 1


# ================= Logistic =================

@patch("optuna.create_study")
def test_logistic_train(mock_create_study):

    class DummyStudy:
        best_params = {"C": 1.0, "solver": "lbfgs"}

        def optimize(self, *args, **kwargs):
            pass  # fake optuna optimization

    mock_create_study.return_value = DummyStudy()

    X = pd.DataFrame(np.random.rand(50, 5))
    y = pd.Series(np.random.randint(0, 2, 50))

    model = LogisticRegressionModel(X, y, X, y, X, y)
    model.model_train()

    assert model.model is not None