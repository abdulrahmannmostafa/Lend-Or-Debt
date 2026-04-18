import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
#Third-party libraries
import warnings
warnings.filterwarnings("ignore")
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
def test_loading_checker_transform():    
    train_dataset, val_dataset,test_dataset,full_df=eda.load_data_transformed()
    expected = len(train_dataset) + len(val_dataset) + len(test_dataset)
    assert len(full_df) == expected
def test_loading_checker_cleaned():    
    train_dataset, val_dataset,test_dataset,full_df=eda.load_data_cleaned()
    expected = len(train_dataset) + len(val_dataset) + len(test_dataset)
    assert len(full_df) == expected    
    