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
    assert not scaled_data.empty, "empty data is not allowed" 
    assert scaled_data.isnull().sum().sum() == 0, "Nulls shouldn't exist in data"
    if 'SEX' in scaled_data.columns:
        assert scaled_data['SEX'].isin([0, 1]).all() ,"label shift must be applied to be binary"

def test_pearson_sorted(extractor, scaled_data):
    X = scaled_data.drop(columns=['default payment next month'])
    y = scaled_data['default payment next month']

    _, scores = extractor.pearson_correlation(X, y, k=5)

    values = scores['Score'].values
    assert all(values[i] >= values[i+1] for i in range(len(values)-1)),"order must be descending" # sorted des

def test_no_duplicate_selected_features(extractor, scaled_data):
    X = scaled_data.drop(columns=['default payment next month'])
    y = scaled_data['default payment next month']

    selected, _ = extractor.pearson_correlation(X, y, k=5)

    assert len(selected) == len(set(selected)),"columns must be unique"    

def test_spearman_selection(extractor, scaled_data):
    X = scaled_data.drop(columns=['default payment next month'])
    y = scaled_data['default payment next month']    
    k = 5
    selected_features, feature_scores = extractor.spearman_correlation(X, y, k=k)
    
    assert len(selected_features) == k,f"should be {k} features only selected "
    assert len(feature_scores) >= k," must be positive"

def test_mutual_info_selection(extractor, scaled_data):
    X = scaled_data.drop(columns=['default payment next month'])
    y = scaled_data['default payment next month']    
    k = 5
    selected_features, feature_scores = extractor.select_k_best_mutual_info(X, y, k=k)    
    assert len(selected_features) == k,f"should be {k} features only selected "
    assert feature_scores['Score'].min() >= 0," must be positive"
    assert feature_scores["Score"].sum() > 0,"data is meaning less with zero info"

def test_selection_methods_keys(extractor, scaled_data):
    X = scaled_data.drop(columns=['default payment next month'])
    y = scaled_data['default payment next month']

    results = extractor.test_feature_selection_methods(X, y, k=5)
    assert set(results.keys()) == {"mutual_info", "pearson", "spearman"},"Those 3 keys must exist"

def test_selection_methods_structure(extractor, scaled_data):
    X = scaled_data.drop(columns=['default payment next month'])
    y = scaled_data['default payment next month']

    results = extractor.test_feature_selection_methods(X, y, k=5)

    for method, val in results.items():
        assert "cols" in val
        assert "scores" in val
        
def test_pca_after_scaling(extractor, scaled_data):
    X = scaled_data.drop(columns=['default payment next month'])
    X_pca, _, _ = extractor.apply_pca(X, X, X, n_components=2)
    
    assert X_pca.shape[1] == 2,"Two components only to be selected"

def test_no_data_leakage(extractor, scaled_data):
    half = len(scaled_data) // 2
    train = scaled_data.iloc[:half]
    test  = scaled_data.iloc[half:]

    extractor.smart_scaler(train, flag=True)
    transformed_test = extractor.smart_scaler(test, flag=False)

    assert transformed_test.shape[0] == test.shape[0]

def test_no_duplicate_features(scaled_data):
    assert len(scaled_data.columns) == len(set(scaled_data.columns))        


def test_row_count_preserved(extractor):
    df = pd.read_csv(train_transformed_path).head(50)
    result = extractor.smart_scaler(df, flag=True)

    assert result.shape[0] == df.shape[0]

def test_scaler_returns_dataframe(extractor):
    df = pd.read_csv(train_transformed_path).head(50)
    result = extractor.smart_scaler(df, flag=True)

    assert isinstance(result, pd.DataFrame),"must be dataframe"

def test_marriage_one_hot(extractor):
    df = pd.read_csv(train_transformed_path).head(50)
    result = extractor.smart_scaler(df, flag=True)

    assert "MARRIAGE" not in result.columns

    marriage_cols = [c for c in result.columns if c.startswith("MARRIAGE_")]
    assert len(marriage_cols) > 0   


def test_missing_column_raises_error(extractor):
    df = pd.read_csv(train_transformed_path).head(50)

    df = df.drop(columns=["PAY_AMT1"], errors="ignore")

    with pytest.raises(Exception):
        extractor.smart_scaler(df, flag=True)
        

def test_scaling_changes_values(extractor):
    df = pd.read_csv(train_transformed_path).head(50)

    before = df["PAY_AMT1"].copy()
    after = extractor.smart_scaler(df, flag=True)

    assert not np.allclose(before.values, after["PAY_AMT1"].values),"No scaling happened"

def test_pearson_top_feature_has_max_score(extractor, scaled_data):
    X = scaled_data.drop(columns=['default payment next month'])
    y = scaled_data['default payment next month']
    _, scores = extractor.pearson_correlation(X, y, k=5)
    assert scores.iloc[0]["Score"] == scores["Score"].max()

def test_mutual_info_features_valid(extractor, scaled_data):
    X = scaled_data.drop(columns=['default payment next month'])
    y = scaled_data['default payment next month']

    selected, _ = extractor.select_k_best_mutual_info(X, y, k=5)

    for col in selected:
        assert col in X.columns
               
    

def test_pca_consistent_components(extractor, scaled_data):
    X = scaled_data.drop(columns=['default payment next month'])

    X1, _, _ = extractor.apply_pca(X, X, X, n_components=2)
    X2, _, _ = extractor.apply_pca(X, X, X, n_components=2)

    assert X1.shape == X2.shape                        