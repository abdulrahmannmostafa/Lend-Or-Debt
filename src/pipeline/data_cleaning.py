import numpy as np
import pandas as pd
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)

TARGET_COL = "default payment next month"

BILL_AMT_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_AMT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]

# Columns with positive skew confirmed by KS-test failures in Phase 1 report
LOG1P_COLS = PAY_AMT_COLS + BILL_AMT_COLS + ["avg_bill", "avg_payment"]

POWER_TRANSFORM_COLS = ["LIMIT_BAL", "AGE"]


def type_coercion(df: pd.DataFrame) -> pd.DataFrame:
    """
    change type of LOG1P_COLS to float.

    """
    df.rename(columns={"PAY_0": "PAY_1"}, inplace=True)
    for col in LOG1P_COLS:
        if col in df.columns:
            df[col] = df[col].astype(float)
    return df


def handle_outliers(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Two-stage outlier treatment for financial amount columns.

         Capping (fit on train only):
        For each PAY_AMTi/BILL_AMTi pair, compute the 99th-percentile cap
        from the TRAINING split only, then apply that same cap to val/test.

    """
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    # ---  compute caps from train, apply to all splits ---
    log.info("Outliers  computing PAY/BILL ratio caps from TRAIN split only ...")
    for i in range(1, 7):
        pay_col = f"PAY_AMT{i}"
        bill_col = f"BILL_AMT{i}"

        if pay_col not in train_df.columns or bill_col not in train_df.columns:
            continue

        positive_bill_mask_train = train_df[bill_col] > 0
        if not positive_bill_mask_train.any():
            continue

        # FIT: compute 99th-pct cap from training rows only
        ratio_train = (
            train_df.loc[positive_bill_mask_train, pay_col]
            / train_df.loc[positive_bill_mask_train, bill_col]
        )
        cap = float(np.percentile(ratio_train, 99))

        # APPLY cap to each split
        for df_part, label in [(train_df, "train"), (val_df, "val"), (test_df, "test")]:
            pos_mask = df_part[bill_col] > 0
            if not pos_mask.any():
                continue
            ratio = df_part.loc[pos_mask, pay_col] / df_part.loc[pos_mask, bill_col]
            n_capped = int((ratio > cap).sum())
            df_part.loc[pos_mask, pay_col] = (
                np.minimum(ratio, cap) * df_part.loc[pos_mask, bill_col]
            )
            if n_capped:
                log.info(
                    "  %s/%s [%s]: 99th-pct cap (train) = %.4f | rows capped = %d",
                    pay_col,
                    bill_col,
                    label,
                    cap,
                    n_capped,
                )

    # Recompute avg_payment after capping so engineered features stay consistent
    pay_present = [c for c in PAY_AMT_COLS if c in train_df.columns]
    for df_part, label in [(train_df, "train"), (val_df, "val"), (test_df, "test")]:
        if pay_present and "avg_payment" in df_part.columns:
            df_part["avg_payment"] = df_part[pay_present].mean(axis=1).fillna(0.0)
    log.info("avg_payment recomputed in all splits after capping.")

    return train_df, val_df, test_df


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
        log.info(
            f"Removed {duplicates_removed} duplicate records ({pct:.2f}% of dataset)"
        )
    else:
        log.info("No duplicate records found.")

    return df


def split_dataset(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified train / val / test split.

    Stratification preserves the original 77.88/22.12 class ratio in every
    split, ensuring evaluation metrics are comparable across phases.

    Returns:
        (train_df, val_df, test_df) — target column included in each.
    """

    test_frac = round(1.0 - train_frac - val_frac, 10)
    assert test_frac > 0, "train_frac + val_frac must be < 1.0"

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # First split: train+val vs test
    X_tv, X_test, y_tv, y_test = train_test_split(
        X,
        y,
        test_size=test_frac,
        stratify=y,
        random_state=random_state,
    )

    # Second split: train vs val (relative proportion within train+val)
    relative_val = val_frac / (train_frac + val_frac)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv,
        y_tv,
        test_size=relative_val,
        stratify=y_tv,
        random_state=random_state,
    )

    def _recombine(X_part, y_part):
        out = X_part.copy().reset_index(drop=True)
        out[target_col] = y_part.values
        return out

    train_df = _recombine(X_train, y_train)
    val_df = _recombine(X_val, y_val)
    test_df = _recombine(X_test, y_test)

    log.info(
        "Split: train=%d (%.0f%%) | val=%d (%.0f%%) | test=%d (%.0f%%)",
        len(train_df),
        train_frac * 100,
        len(val_df),
        val_frac * 100,
        len(test_df),
        test_frac * 100,
    )
    for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
        rate = part[target_col].mean()
        log.info("  %s default rate: %.2f%%", name, rate * 100)

    return train_df, val_df, test_df


# =========================================================
# CLEANING STEP
# =========================================================


def clean_data(
    input_path: str,
    train_output: str,
    val_output: str,
    test_output: str,
) -> None:
    """
    File-based cleaning pipeline aligned with master_pipeline contract.
    """

    log.info("Starting data cleaning pipeline...")
    df = pd.read_csv(input_path)
    log.info("--- Step 1: Handle Type Coercion ---")
    df = type_coercion(df)
    log.info("--- Step 2: Handle uniqueness (remove duplicates) ---")
    df = handle_uniqueness(df)
    log.info("--- Step 3: Train/Val/Test split (70/15/15, stratified) ---")
    train_df, val_df, test_df = split_dataset(df)
    log.info("--- Step 4: Handle outliers (capping ) ---")
    train_df, val_df, test_df = handle_outliers(train_df, val_df, test_df)

    # save outputs
    Path(train_output).parent.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(train_output, index=False)
    val_df.to_csv(val_output, index=False)
    test_df.to_csv(test_output, index=False)

    log.info("Saved train -> %s", train_output)
    log.info("Saved val   -> %s", val_output)
    log.info("Saved test  -> %s", test_output)


if __name__ == "__main__":
    try:
        if Path("data/taiwan_merged1.csv").exists():
            clean_data(
                input_path="data/taiwan_merged1.csv",
                train_output="data/clean/train_cleaned.csv",
                val_output="data/clean/val_cleaned.csv",
                test_output="data/clean/test_cleaned.csv",
            )
        else:
                log.warning("Input file not found for standalone run.")
    except Exception as _e:
        log.warning("Standalone run failed: %s", _e)

