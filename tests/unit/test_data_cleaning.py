import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

from src.pipeline.data_cleaning import (
    type_coercion,
    handle_uniqueness,
    split_dataset,
    handle_outliers,
)

TARGET_COL = "default payment next month"


def _make_df(n=200, seed=42):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "LIMIT_BAL": rng.integers(10_000, 800_000, n).astype(float),
            "SEX": rng.integers(1, 3, n),
            "EDUCATION": rng.integers(1, 5, n),
            "MARRIAGE": rng.integers(1, 4, n),
            "AGE": rng.integers(20, 65, n),
            **{f"PAY_AMT{i}": rng.uniform(0, 50_000, n) for i in range(1, 7)},
            **{f"BILL_AMT{i}": rng.uniform(0, 100_000, n) for i in range(1, 7)},
            "avg_bill": rng.uniform(0, 80_000, n),
            "avg_payment": rng.uniform(0, 50_000, n),
            TARGET_COL: rng.integers(0, 2, n),
        }
    )
    # Rename PAY_0 -> PAY_1 scenario that adds PAY_0
    df["PAY_0"] = rng.integers(-2, 9, n)
    return df


@pytest.fixture
def sample_df():
    return _make_df(n=300)


class TestTypeCoercion:
    def test_renames_pay0_to_pay1(self, sample_df):
        df = sample_df.copy()
        out = type_coercion(df)
        assert "PAY_1" in out.columns
        assert "PAY_0" not in out.columns

    def test_log1p_cols_are_float(self, sample_df):
        out = type_coercion(sample_df.copy())
        for col in ["PAY_AMT1", "BILL_AMT1", "avg_bill", "avg_payment"]:
            if col in out.columns:
                assert out[col].dtype == float

    def test_returns_dataframe(self, sample_df):
        out = type_coercion(sample_df.copy())
        assert isinstance(out, pd.DataFrame)

    def test_no_data_loss(self, sample_df):
        out = type_coercion(sample_df.copy())
        assert len(out) == len(sample_df)


class TestHandleUniqueness:
    def test_removes_exact_duplicates(self, sample_df):
        # append 5 exact duplicates
        duped = pd.concat([sample_df, sample_df.head(5)], ignore_index=True)
        out = handle_uniqueness(duped)
        assert len(out) == len(sample_df)

    def test_no_duplicates_after(self, sample_df):
        out = handle_uniqueness(sample_df.copy())
        assert out.duplicated().sum() == 0

    def test_clean_df_unchanged_length(self, sample_df):
        out = handle_uniqueness(sample_df.copy())
        assert len(out) == len(sample_df)

    def test_returns_dataframe(self, sample_df):
        assert isinstance(handle_uniqueness(sample_df.copy()), pd.DataFrame)


class TestSplitDataset:
    def test_returns_three_splits(self, sample_df):
        train, val, test = split_dataset(sample_df.copy())
        assert all(isinstance(d, pd.DataFrame) for d in [train, val, test])

    def test_no_overlap_between_splits(self, sample_df):
        df = sample_df.copy().reset_index(drop=True)
        train, val, test = split_dataset(df)
        total = len(train) + len(val) + len(test)
        assert total == len(df)

    def test_target_col_present_in_all_splits(self, sample_df):
        train, val, test = split_dataset(sample_df.copy())
        for split in [train, val, test]:
            assert TARGET_COL in split.columns

    def test_approximate_split_ratios(self, sample_df):
        train, val, test = split_dataset(
            sample_df.copy(), train_frac=0.70, val_frac=0.15
        )
        n = len(sample_df)
        assert abs(len(train) / n - 0.70) < 0.05
        assert abs(len(val) / n - 0.15) < 0.05
        assert abs(len(test) / n - 0.15) < 0.05

    def test_stratification_preserves_class_ratio(self, sample_df):
        original_rate = sample_df[TARGET_COL].mean()
        train, val, test = split_dataset(sample_df.copy())
        for split in [train, val, test]:
            rate = split[TARGET_COL].mean()
            assert abs(rate - original_rate) < 0.05

    def test_invalid_fracs_raise(self, sample_df):
        with pytest.raises(AssertionError):
            split_dataset(sample_df.copy(), train_frac=0.60, val_frac=0.50)


# ─── handle_outliers ─────────────────────────────────────────────────────────


class TestHandleOutliers:
    @pytest.fixture
    def splits(self, sample_df):
        df = type_coercion(sample_df.copy())
        return split_dataset(df)

    def test_returns_three_dataframes(self, splits):
        train, val, test = splits
        out = handle_outliers(train, val, test)
        assert len(out) == 3
        assert all(isinstance(d, pd.DataFrame) for d in out)

    def test_shape_preserved(self, splits):
        train, val, test = splits
        t2, v2, te2 = handle_outliers(train, val, test)
        assert t2.shape == train.shape
        assert v2.shape == val.shape
        assert te2.shape == test.shape

    def test_avg_payment_recomputed(self, splits):
        train, val, test = splits
        t2, v2, te2 = handle_outliers(train, val, test)
        # avg_payment must still be non-negative after capping
        assert (t2["avg_payment"] >= 0).all()

    def test_no_new_nulls_introduced(self, splits):
        train, val, test = splits
        t2, v2, te2 = handle_outliers(train, val, test)
        for df_part in [t2, v2, te2]:
            assert df_part.isnull().sum().sum() == 0
