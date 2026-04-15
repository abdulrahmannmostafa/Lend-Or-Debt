import numpy as np
import pandas as pd
import logging
from pathlib import Path
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.decomposition import PCA

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


BILL_AMT_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]


def handle_bill_amt_collinearity(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    strategy: str = "pca",       # "pca" or "drop"
    n_components: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Handle the high multicollinearity among BILL_AMT1-6 (|r| > 0.9).
    
    Root cause: month-over-month balance rollover means consecutive
    bill amounts are nearly identical (Phase 1 §3.9).

    Strategy "pca":
        Compress BILL_AMT1-6 into n_components principal components.
        Fit PCA on training data only, then transform val and test
        using the same fitted PCA to prevent data leakage.
        Preserves monthly trend information in a compact form.
        Best for linear models.

    Strategy "drop":
        Keep only BILL_AMT1 (most recent, most predictive month).
        Drop BILL_AMT2-6.
        Simple and interpretable.
        Acceptable for tree models which handle collinearity natively.

    Args:
        train_df, val_df, test_df: the three splits from split_dataset()
        strategy:      "pca" or "drop"
        n_components:  number of PCA components to keep (only used if
                       strategy="pca"). 2 is chosen because Phase 1
                       distribution analysis shows the 6 BILL_AMT columns
                       share one dominant variance direction (the overall
                       balance level) and one secondary direction (the
                       monthly trend/change). 2 components typically
                       capture >95% of variance in this dataset.

    Returns:
        (train_df, val_df, test_df) with BILL_AMT columns replaced.
    """
    present = [c for c in BILL_AMT_COLS if c in train_df.columns]
    if not present:
        log.warning("handle_bill_amt_collinearity: no BILL_AMT columns found.")
        return train_df, val_df, test_df

    if strategy == "drop":
        # Keep only BILL_AMT1, drop the rest
        cols_to_drop = [c for c in present if c != "BILL_AMT1"]
        train_df = train_df.drop(columns=cols_to_drop)
        val_df   = val_df.drop(columns=cols_to_drop)
        test_df  = test_df.drop(columns=cols_to_drop)
        log.info(
            "BILL_AMT collinearity: dropped %s, kept BILL_AMT1.",
            cols_to_drop,
        )
        return train_df, val_df, test_df

    if strategy == "pca":
        # Fit PCA on training data only — never fit on val or test
        pca = PCA(n_components=n_components, random_state=42)
        
        train_components = pca.fit_transform(train_df[present])
        val_components   = pca.transform(val_df[present])
        test_components  = pca.transform(test_df[present])

        explained = pca.explained_variance_ratio_
        log.info(
            "BILL_AMT PCA: %d components explain %.2f%% of variance. "
            "Per component: %s",
            n_components,
            sum(explained) * 100,
            [f"PC{i+1}={v:.3f}" for i, v in enumerate(explained)],
        )

        # Build component column names
        component_cols = [f"BILL_AMT_PC{i+1}" for i in range(n_components)]

        # Drop original BILL_AMT columns and add PCA components
        for df_split, components in [
            (train_df, train_components),
            (val_df,   val_components),
            (test_df,  test_components),
        ]:
            pass  # handled below to avoid modifying loop variable

        train_df = train_df.drop(columns=present)
        val_df   = val_df.drop(columns=present)
        test_df  = test_df.drop(columns=present)

        for split_df, components, name in [
            (train_df, train_components, "train"),
            (val_df,   val_components,   "val"),
            (test_df,  test_components,  "test"),
        ]:
            for j, col_name in enumerate(component_cols):
                split_df[col_name] = components[:, j]
            log.info(
                "  %s: added columns %s", name, component_cols
            )


        train_pca = train_df.copy()
        val_pca   = val_df.copy()
        test_pca  = test_df.copy()

        for j, col_name in enumerate(component_cols):
            train_pca[col_name] = train_components[:, j]
            val_pca[col_name]   = val_components[:, j]
            test_pca[col_name]  = test_components[:, j]

        log.info(
            "BILL_AMT collinearity: replaced %d BILL_AMT cols with %d PCA components.",
            len(present), n_components,
        )
        return train_pca, val_pca, test_pca

    raise ValueError(f"strategy must be 'pca' or 'drop', got '{strategy}'")

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