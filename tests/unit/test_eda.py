import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
#Third-party libraries
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

@pytest.fixture(autouse=True)
def close_plots():
    yield
    plt.close("all")

#Local
from src.pipeline.eda import EDA
from src.pipeline.config import (
    train_cleaned_path,
    val_cleaned_path,
    test_cleaned_path,
    train_transformed_path,
    val_transformed_path,
    test_transformed_path,
)
eda = EDA(
    train_input_cleaned=train_cleaned_path,
    val_input_cleaned=val_cleaned_path,
    test_input_cleaned=test_cleaned_path,
    train_input_transformed=train_transformed_path,
    val_input_transformed=val_transformed_path,
    test_input_transformed=test_transformed_path
)
expected_values = {
    "SEX": {1, 2},
    "MARRIAGE": {1, 2, 3},
    "EDUCATION": {1, 2, 3, 4},
    "default payment next month": {0, 1},
    "is_anomaly": {0, 1},
    "is_underpaying": {0, 1},
}

@pytest.fixture(scope="module")
def transformed_df():
    _, _, _, df = eda.load_data_transformed()
    return df

def test_loading_checker_transform():    
    train_dataset, val_dataset,test_dataset,full_df=eda.load_data_transformed()
    expected = len(train_dataset) + len(val_dataset) + len(test_dataset)
    assert len(full_df) == expected
    
def test_loading_checker_cleaned():    
    train_dataset, val_dataset,test_dataset,full_df=eda.load_data_cleaned()
    expected = len(train_dataset) + len(val_dataset) + len(test_dataset)
    assert len(full_df) == expected    
    
def test_loading_checker_transform_cleaned():    
    train_dataset, val_dataset,test_dataset,full_df=eda.load_data_transformed_without_smote()
    expected = len(train_dataset) + len(val_dataset) + len(test_dataset)
    assert len(full_df) == expected  
    
def test_dataset_not_empty(transformed_df):
    assert len(transformed_df) > 0
        
def test_continuous_columns_exist(transformed_df):   
    for col in eda.continuous_features:
        assert col in transformed_df.columns  
         
def test_discrete_columns_exist(transformed_df):
    for col in eda.discrete_features:
        assert col in transformed_df.columns  
          
def test_mapping_validity():
    for feature, labels in eda.mapping.items():
        assert isinstance(labels, (list, dict))
        assert len(labels) > 0

def test_discrete_values_valid(transformed_df):
    for feature, valid_values in expected_values.items():
        actual = set(transformed_df[feature].dropna().unique())
        assert actual.issubset(valid_values)                 
        
def test_no_nan_in_target(transformed_df):
    assert transformed_df["default payment next month"].isna().sum() == 0             
    
    
def test_numeric_features(transformed_df):
    for col in eda.continuous_features:
        assert pd.api.types.is_numeric_dtype(transformed_df[col])  
        
def test_univariate_runs(transformed_df):
    eda.full_df_transformed = transformed_df
    eda.apply_univariate("LIMIT_BAL")

def test_pie_chart_runs(transformed_df):
    eda.full_df_transformed = transformed_df
    mapping_dict = dict(enumerate(eda.mapping["SEX"]))
    eda.apply_pie_chart("SEX", mapping_dict)

def test_continuous_vs_continuous_runs(transformed_df):
    eda.full_df_transformed = transformed_df
    eda.continuous_vs_continuous_eda("LIMIT_BAL", "AGE")

def test_continuous_vs_discrete_runs(transformed_df):
    eda.full_df_transformed = transformed_df
    eda.continuous_versus_discrete_eda("LIMIT_BAL")

def test_discrete_vs_target_runs(transformed_df):
    eda.full_df_transformed = transformed_df
    eda.discrete_versus_target_stacked("default payment next month")

def test_dashboard_after_smote_runs(transformed_df):
    eda.full_df_transformed = transformed_df
    eda.draw_dashboard_after_smote()

def test_dashboard_before_smote_runs(transformed_df):
    eda.full_df_transformed_without_smote = transformed_df
    eda.draw_dashboard_before_smote()
        
def test_correlation_matrix_shape(transformed_df):    
    corr = transformed_df.corr()
    assert corr.shape[0] == corr.shape[1]  

def test_no_duplicate_target(transformed_df): 
    assert transformed_df.columns.tolist().count("default payment next month") == 1    
    
def test_target_values(transformed_df):
    assert set(transformed_df["default payment next month"].unique()) <= {0, 1}                 

def test_target_balance(transformed_df):
    counts = transformed_df["default payment next month"].value_counts()
    ratio = counts.min() / counts.max()
    assert ratio > 0.7    