import numpy as np
import pandas as pd
import logging
from pathlib import Path
from sklearn.preprocessing import StandardScaler, PowerTransformer
from imblearn.over_sampling import SMOTE


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)



def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cap extreme values at the 99th percentile to reduce the influence of outliers.
    and handle skewness in payment and bill amounts.
    """
    log.info("Capping extreme values at 99th percentile...")

    for i in range(1, 7):
        pay_col = f"PAY_AMT{i}"
        bill_col = f"BILL_AMT{i}"

        if pay_col in df.columns and bill_col in df.columns:
            # Cap extreme PAY/BILL ratios at the 99th percentile
            mask = df[bill_col] > 0
            if mask.any():
                ratio = df.loc[mask, pay_col] / df.loc[mask, bill_col]
                cap = np.percentile(ratio, 99)
                capped_ratio = np.minimum(ratio, cap)
                df.loc[mask, pay_col] = capped_ratio * df.loc[mask, bill_col]
                log.info(f"{pay_col}/{bill_col} capped at 99th percentile: {cap:.2f}")

    df["PAY_AMT1"] = np.log1p(df["PAY_AMT1"])
    df["BILL_AMT"] = np.log1p(df["BILL_AMT"])
    log.info("Applied log transformation to PAY_AMT1 and BILL_AMT to handle outliers and reduce skewness.")


    skewed_cols = [col for col in ["LIMIT_BAL", "AGE"] if col in df.columns]

    if not skewed_cols:
        log.info("No demographic skewed columns found for power transform.")
        return df

    # Yeo-Johnson handles zero/negative values safely.
    transformer = PowerTransformer(method="yeo-johnson", standardize=True)
    df[skewed_cols] = transformer.fit_transform(df[skewed_cols])
    log.info(f"Applied PowerTransformer to: {', '.join(skewed_cols)}")
    return df



def handle_uniqueness(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove exact duplicate records across all columns.
    """
    log.info("Removing duplicate records...")

    before_count = len(df)
    df = df.drop_duplicates(keep="first")
    after_count = len(df)
    duplicates_removed = before_count - after_count

    if duplicates_removed > 0:
        pct = (duplicates_removed / before_count) * 100
        log.info(f"Removed {duplicates_removed} duplicate records ({pct:.2f}% of dataset)")
    else:
        log.info("No duplicate records found.")

    return df


def handle_data_imbalance(
    train_df: pd.DataFrame,
    target_col: str,
    random_state: int = 42,
    k_neighbors: int = 5,
) -> pd.DataFrame:
    """
    Apply SMOTE directly on an already prepared training dataframe.
    """
    log.info("Applying SMOTE on provided training dataframe...")

    if target_col not in train_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]

    smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    # Preserve pandas structures for downstream feature engineering/modeling.
    X_resampled = pd.DataFrame(X_resampled, columns=X_train.columns)
    y_name = y_train.name if hasattr(y_train, "name") and y_train.name else "target"
    y_resampled = pd.Series(y_resampled, name=y_name)
    resampled_df = X_resampled.copy()
    resampled_df[y_name] = y_resampled

    before_counts = y_train.value_counts().to_dict()
    after_counts = y_resampled.value_counts().to_dict()
    log.info(f"Class distribution before SMOTE: {before_counts}")
    log.info(f"Class distribution after SMOTE: {after_counts}")

    return resampled_df




# =========================================================
# CLEANING STEP
# =========================================================

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main cleaning pipeline aligned with validation checks
    """
    df = df.copy()
    df = handle_outliers(df)
    df = handle_uniqueness(df)
    df = handle_data_imbalance(df)

    return df

if __name__ == "__main__":
    clean_data()

if __name__ != "__main__":
    try:
        if Path("data/taiwan_merged.csv").exists():
            clean_data()
    except Exception as e:
        log.warning(f"Auto-validation failed during import: {e}")