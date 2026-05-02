import pytest
import pandas as pd
import numpy as np
import sys
import os
import warnings
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from src.pipeline.config import train_transformed_path
from src.pipeline.feature_extraction import FeatureExtractor 

@pytest.fixture
def extractor():
    return FeatureExtractor()

@pytest.fixture
def scaled_data(extractor):
    warnings.filterwarnings("ignore", category=UserWarning)
    if not os.path.exists(train_transformed_path):
        pytest.skip(f"Missing data: {train_transformed_path}")
    
    df = pd.read_csv(train_transformed_path).head(100)
    df_scaled = extractor.smart_scaler(df, flag=True)
    
    return df_scaled

def test_smart_scaler_outputs(scaled_data):
    assert not scaled_data.empty
    assert scaled_data.isnull().sum().sum() == 0
    if 'SEX' in scaled_data.columns:
        assert scaled_data['SEX'].isin([0, 1]).all()

def test_pearson_selection(extractor, scaled_data):
    X = scaled_data.drop(columns=['default payment next month'])
    y = scaled_data['default payment next month']
    
    k = 5
    selected_features, feature_scores = extractor.pearson_correlation(X, y, k=k)
    
    assert len(selected_features) == k
    assert "Feature" in feature_scores.columns
    assert "Score" in feature_scores.columns
    assert feature_scores['Score'].iloc[0] >= feature_scores['Score'].iloc[-1] # sorted des

def test_spearman_selection(extractor, scaled_data):
    X = scaled_data.drop(columns=['default payment next month'])
    y = scaled_data['default payment next month']
    
    k = 5
    selected_features, feature_scores = extractor.spearman_correlation(X, y, k=k)
    
    assert len(selected_features) == k
    assert len(feature_scores) >= k

def test_mutual_info_selection(extractor, scaled_data):
    X = scaled_data.drop(columns=['default payment next month'])
    y = scaled_data['default payment next month']
    
    k = 5
    selected_features, feature_scores = extractor.select_k_best_mutual_info(X, y, k=k)
    
    assert len(selected_features) == k
    assert feature_scores['Score'].min() >= 0

def test_all_selection_methods_combined(extractor, scaled_data):
    X = scaled_data.drop(columns=['default payment next month'])
    y = scaled_data['default payment next month']
    
    k = 10
    results = extractor.test_feature_selection_methods(X, y, k=k)
    
    assert "mutual_info" in results
    assert "pearson" in results
    assert "spearman" in results
    assert len(results["mutual_info"]["cols"]) == k

def test_pca_after_scaling(extractor, scaled_data):
    X = scaled_data.drop(columns=['default payment next month'])
    X_pca, _, _ = extractor.apply_pca(X, X, X, n_components=2)
    
    assert X_pca.shape[1] == 2
