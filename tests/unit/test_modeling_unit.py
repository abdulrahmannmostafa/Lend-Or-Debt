import pytest
import pandas as pd
import numpy as np
import sys
import os
from unittest.mock import patch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from src.pipeline.mlflow.models import BaseModel, LogisticRegressionModel
from src.pipeline.mlflow.model_selection_evalutions import DataLoader


@pytest.fixture
def data_loader():
    return DataLoader()


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
    dl.run_experiment(
        model_type=6,
        smote=False,
        feature_selection="spearman",
        version=1,
        k=5
    )

    assert mock_run.called


@patch.object(DataLoader, "run_dt")
def test_selected_features_used(mock_run):
    dl = DataLoader()
    dl.run_experiment(
        model_type=6,
        smote=False,
        feature_selection="spearman",
        version=1,
        k=5
    )

    assert dl.X_train.shape[1] == 5


@patch.object(DataLoader, "run_dt")
def test_feature_consistency_between_sets_features(mock_run):
    dl = DataLoader()
    dl.run_experiment(
        model_type=6,
        smote=False,
        feature_selection="spearman",
        version=1,
        k=5
    )

    assert list(dl.X_train.columns) == list(dl.X_val.columns)
    assert list(dl.X_train.columns) == list(dl.X_test.columns)


@patch.object(DataLoader, "run_dt")
def test_intersect_feature_selection(mock_run):
    dl = DataLoader()
    dl.run_experiment(
        model_type=6,
        smote=False,
        feature_selection="intersect",
        version=1,
        k=5
    )

    assert dl.X_train.shape[1] <= 5,"features must be <= chosen number for both"


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



# MLflow

@patch("mlflow.log_metric")
@patch("mlflow.log_params") 
@patch("mlflow.set_tag")
@patch("mlflow.start_run")
def test_mlflow_tags(mock_start_run, mock_set_tag, mock_log_params, mock_log_metric):
    dl = DataLoader()

    dl.run_experiment(
        model_type=6,
        smote=False,
        feature_selection="spearman",
        version=2,
        k=5
    )

    assert mock_set_tag.called


# BaseModel logic

def test_find_best_threshold():
    X = pd.DataFrame(np.random.rand(20, 3))
    y = pd.Series(np.random.randint(0, 2, 20))
    model = BaseModel(X, y, X, y, X, y)
    proba = np.random.rand(20)
    mcc, t = model.find_best_threshold_mcc(proba, y)

    assert 0 <= t <= 1,"out of range"



# Logistic model

@patch("mlflow.log_param")
@patch("mlflow.start_run")
@patch("optuna.create_study")
def test_logistic_train(mock_create_study, mock_start_run, mock_log_param):
    class DummyStudy:
        best_params = {"C": 1.0, "solver": "lbfgs"}

        def optimize(self, *args, **kwargs):
            pass

    mock_create_study.return_value = DummyStudy()

    X = pd.DataFrame(np.random.rand(50, 5))
    y = pd.Series(np.random.randint(0, 2, 50))

    model = LogisticRegressionModel(X, y, X, y, X, y)
    model.model_train()

    assert model.model is not None