import numpy as np
import pandas as pd
import logging
from pathlib import Path
from sklearn.preprocessing import PowerTransformer, StandardScaler


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def standardize_macro_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize macroeconomic features for numerical stability.
    """
    macro_cols = [col for col in ["GDP", "CPI", "TAIEX"] if col in df.columns]

    if not macro_cols:
        log.info("No macro features found for standardization.")
        return df

    scaler = StandardScaler()
    df[macro_cols] = scaler.fit_transform(df[macro_cols])
    log.info(f"Standardized macro features: {', '.join(macro_cols)}")

    return df


def add_nonlinear_relationship_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add features for known non-linear payment/utilization relationships.
    """
    return df


def drop_highly_correlated_bill_amounts(df: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
    """
    Drop highly correlated BILL_AMT columns to reduce redundancy.
    """
    bill_cols = [f"BILL_AMT{i}" for i in range(1, 7) if f"BILL_AMT{i}" in df.columns]

    if len(bill_cols) < 2:
        log.info("Not enough BILL_AMT columns to evaluate correlation.")
        return df

    corr_matrix = df[bill_cols].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]

    if to_drop:
        df = df.drop(columns=to_drop)
        log.info(f"Dropped highly correlated BILL_AMT columns (|r| > {threshold}): {', '.join(to_drop)}")
    else:
        log.info(f"No BILL_AMT columns exceeded correlation threshold |r| > {threshold}.")

    return df






def run_transformation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main transformation pipeline
    """

    df = df.copy()
    df = standardize_macro_features(df)
    df = add_nonlinear_relationship_features(df)
    df = drop_highly_correlated_bill_amounts(df, threshold=0.9)

    return df

if __name__ == "__main__":
    run_transformation()

if __name__ != "__main__":
    try:
        if Path("data/taiwan_merged.csv").exists():
            run_transformation()
    except Exception as e:
        log.warning(f"Auto-transformation failed during import: {e}")