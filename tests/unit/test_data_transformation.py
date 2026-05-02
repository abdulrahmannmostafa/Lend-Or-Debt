import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

from src.pipeline.data_transformation import (
    remap_categorical_codes,
    recode_pay_status,
    engineer_credit_features,
    engineer_payment_features,
    engineer_macro_interactions,
    handle_bill_amt_collinearity,
    handle_distribution,
    flag_anomalies,
    handle_data_imbalance,
    standardize_macro_features,
)

TARGET_COL = "default payment next month"


def _base_df(n=200, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "LIMIT_BAL": rng.uniform(10_000, 800_000, n),
            "AGE": rng.uniform(20, 65, n),
            "SEX": rng.integers(1, 3, n),
            "EDUCATION": rng.choice([0, 1, 2, 3, 4, 5, 6], n),
            "MARRIAGE": rng.choice([0, 1, 2, 3], n),
            **{f"PAY_AMT{i}": rng.uniform(0, 30_000, n) for i in range(1, 7)},
            **{f"BILL_AMT{i}": rng.uniform(0, 80_000, n) for i in range(1, 7)},
            **{f"PAY_{i}": rng.integers(-2, 9, n) for i in range(1, 7)},
            "avg_bill": rng.uniform(0, 70_000, n),
            "avg_payment": rng.uniform(0, 30_000, n),
            "total_delinquency": rng.integers(0, 20, n).astype(float),
            "max_delinquency": rng.integers(0, 8, n).astype(float),
            "avg_macro_CPI": np.full(n, 104.5),
            "avg_macro_GDP": np.full(n, 12500.0),
            "avg_macro_TAIEX": np.full(n, 6200.0),
            "avg_macro_rate": np.full(n, 2.0625),
            "avg_macro_unemp": np.full(n, 4.09),
            TARGET_COL: rng.integers(0, 2, n),
        }
    )
    return df


def _three_splits(n=300):
    df = _base_df(n=n)
    train = df.iloc[: int(n * 0.7)].copy().reset_index(drop=True)
    val = df.iloc[int(n * 0.7) : int(n * 0.85)].copy().reset_index(drop=True)
    test = df.iloc[int(n * 0.85) :].copy().reset_index(drop=True)
    return train, val, test


class TestRemapCategoricalCodes:
    def test_education_undoc_codes_remapped(self):
        df = pd.DataFrame({"EDUCATION": [0, 1, 2, 3, 4, 5, 6]})
        out = remap_categorical_codes(df)
        assert set(out["EDUCATION"]).issubset({1, 2, 3, 4})

    def test_marriage_zero_remapped(self):
        df = pd.DataFrame({"MARRIAGE": [0, 1, 2, 3]})
        out = remap_categorical_codes(df)
        assert 0 not in out["MARRIAGE"].values

    def test_valid_codes_unchanged(self):
        df = pd.DataFrame({"EDUCATION": [1, 2, 3, 4], "MARRIAGE": [1, 2, 3, 1]})
        out = remap_categorical_codes(df)
        assert list(out["EDUCATION"]) == [1, 2, 3, 4]
        assert list(out["MARRIAGE"]) == [1, 2, 3, 1]

    def test_returns_dataframe(self):
        assert isinstance(remap_categorical_codes(_base_df()), pd.DataFrame)


class TestRecodePayStatus:
    def test_negative_values_become_zero(self):
        df = pd.DataFrame({"PAY_1": [-2, -1, 0, 1, 2]})
        out = recode_pay_status(df)
        assert out["PAY_1"].min() == 0

    def test_zero_becomes_one(self):
        df = pd.DataFrame({"PAY_1": [0]})
        out = recode_pay_status(df)
        assert out["PAY_1"].iloc[0] == 1

    def test_positive_k_becomes_k_plus_one(self):
        df = pd.DataFrame({"PAY_1": [2, 3, 8]})
        out = recode_pay_status(df)
        assert list(out["PAY_1"]) == [3, 4, 9]

    def test_no_negative_values_after(self):
        df = _base_df()
        out = recode_pay_status(df)
        pay_cols = [c for c in out.columns if c.startswith("PAY_") and len(c) == 5]
        for col in pay_cols:
            assert (out[col] >= 0).all()


class TestEngineerCreditFeatures:
    def test_new_columns_created(self):
        df = _base_df()
        out = engineer_credit_features(df)
        for col in ["utilisation_ratio", "LIMIT_BAL_sq", "limit_x_bill"]:
            assert col in out.columns

    def test_utilisation_ratio_non_negative(self):
        df = _base_df()
        out = engineer_credit_features(df)
        assert (out["utilisation_ratio"] >= 0).all()

    def test_limit_bal_sq_equals_square(self):
        df = _base_df(n=10)
        out = engineer_credit_features(df)
        expected = df["LIMIT_BAL"] ** 2
        pd.testing.assert_series_equal(out["LIMIT_BAL_sq"], expected, check_names=False)

    def test_missing_cols_no_crash(self):
        df = pd.DataFrame({"OTHER": [1, 2, 3]})
        out = engineer_credit_features(df)
        assert "utilisation_ratio" not in out.columns


class TestEngineerPaymentFeatures:
    def test_new_columns_created(self):
        df = _base_df()
        out = engineer_payment_features(df)
        for col in ["payment_ratio", "avg_unpaid_balance", "is_underpaying"]:
            assert col in out.columns

    def test_is_underpaying_binary(self):
        df = _base_df()
        out = engineer_payment_features(df)
        assert set(out["is_underpaying"].unique()).issubset({0, 1})

    def test_payment_ratio_non_negative_when_non_negative_inputs(self):
        df = _base_df()
        out = engineer_payment_features(df)
        assert (out["payment_ratio"] >= 0).all()


class TestEngineerMacroInteractions:
    def test_raw_macro_cols_dropped(self):
        df = engineer_payment_features(engineer_credit_features(_base_df()))
        out = engineer_macro_interactions(df)
        for col in [
            "avg_macro_CPI",
            "avg_macro_GDP",
            "avg_macro_TAIEX",
            "avg_macro_rate",
            "avg_macro_unemp",
        ]:
            assert col not in out.columns

    def test_interaction_cols_created(self):
        df = engineer_payment_features(engineer_credit_features(_base_df()))
        out = engineer_macro_interactions(df)
        for col in [
            "cpi_risk_norm",
            "gdp_x_payment_ratio",
            "taiex_x_utilisation",
            "rate_x_delinquency",
            "unemp_x_delinquency",
        ]:
            assert col in out.columns

    def test_no_macro_no_crash(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        out = engineer_macro_interactions(df)
        assert list(out.columns) == ["a", "b"]


class TestHandleBillAmtCollinearity:
    def test_pca_strategy_replaces_cols(self):
        train, val, test = _three_splits()
        t2, v2, te2 = handle_bill_amt_collinearity(
            train, val, test, strategy="pca", n_components=2
        )
        assert "BILL_AMT_PC1" in t2.columns
        assert "BILL_AMT1" not in t2.columns

    def test_drop_strategy_keeps_only_bill1(self):
        train, val, test = _three_splits()
        t2, v2, te2 = handle_bill_amt_collinearity(train, val, test, strategy="drop")
        assert "BILL_AMT1" in t2.columns
        for i in range(2, 7):
            assert f"BILL_AMT{i}" not in t2.columns

    def test_pca_applied_to_all_splits(self):
        train, val, test = _three_splits()
        t2, v2, te2 = handle_bill_amt_collinearity(
            train, val, test, strategy="pca", n_components=2
        )
        for df_part in [t2, v2, te2]:
            assert "BILL_AMT_PC1" in df_part.columns

    def test_invalid_strategy_raises(self):
        train, val, test = _three_splits()
        with pytest.raises(ValueError):
            handle_bill_amt_collinearity(train, val, test, strategy="bad")

    def test_row_count_unchanged(self):
        train, val, test = _three_splits()
        t2, v2, te2 = handle_bill_amt_collinearity(train, val, test, strategy="pca")
        assert len(t2) == len(train)


class TestHandleDistribution:
    def test_returns_three_dfs(self):
        train, val, test = _three_splits()
        out = handle_distribution(train, val, test)
        assert len(out) == 3

    def test_shape_unchanged(self):
        train, val, test = _three_splits()
        t2, v2, te2 = handle_distribution(train, val, test)
        assert t2.shape == train.shape

    def test_log1p_cols_transformed(self):
        train, val, test = _three_splits()
        t2, _, _ = handle_distribution(train, val, test)
        # After log1p, values in transformed PAY_AMT cols should be in a smaller range
        assert t2["PAY_AMT1"].max() < train["PAY_AMT1"].max()

    def test_no_nulls_after(self):
        train, val, test = _three_splits()
        t2, v2, te2 = handle_distribution(train, val, test)
        for df_part in [t2, v2, te2]:
            assert df_part.isnull().sum().sum() == 0


class TestFlagAnomalies:
    def test_is_anomaly_column_added(self):
        train, val, test = _three_splits()
        t2, v2, te2 = flag_anomalies(train, val, test)
        for df_part in [t2, v2, te2]:
            assert "is_anomaly" in df_part.columns

    def test_is_anomaly_binary(self):
        train, val, test = _three_splits()
        t2, v2, te2 = flag_anomalies(train, val, test)
        for df_part in [t2, v2, te2]:
            assert set(df_part["is_anomaly"].unique()).issubset({0, 1})

    def test_contamination_rate_approx(self):
        train, val, test = _three_splits(n=500)
        t2, _, _ = flag_anomalies(train, val, test, contamination=0.05)
        rate = t2["is_anomaly"].mean()
        assert 0.02 <= rate <= 0.10  # allow some tolerance

    def test_row_count_unchanged(self):
        train, val, test = _three_splits()
        t2, v2, te2 = flag_anomalies(train, val, test)
        assert len(t2) == len(train)


class TestHandleDataImbalance:
    def test_returns_dataframe(self):
        train, _, _ = _three_splits(n=400)
        out = handle_data_imbalance(train)
        assert isinstance(out, pd.DataFrame)

    def test_classes_balanced_after_smote(self):
        train, _, _ = _three_splits(n=400)
        out = handle_data_imbalance(train)
        counts = out[TARGET_COL].value_counts()
        assert counts[0] == counts[1]

    def test_target_col_present_after(self):
        train, _, _ = _three_splits(n=400)
        out = handle_data_imbalance(train)
        assert TARGET_COL in out.columns

    def test_missing_target_raises(self):
        train, _, _ = _three_splits(n=400)
        train_no_target = train.drop(columns=[TARGET_COL])
        with pytest.raises(ValueError):
            handle_data_imbalance(train_no_target)


class TestStandardizeMacroFeatures:
    def _prep_splits_with_interactions(self):
        train, val, test = _three_splits()
        for df_part in [train, val, test]:
            df_part["cpi_risk_norm"] = np.random.randn(len(df_part))
            df_part["gdp_x_payment_ratio"] = np.random.randn(len(df_part))
        return train, val, test

    def test_returns_three_dfs(self):
        train, val, test = self._prep_splits_with_interactions()
        out = standardize_macro_features(train, val, test)
        assert len(out) == 3

    def test_train_mean_approx_zero(self):
        train, val, test = self._prep_splits_with_interactions()
        t2, _, _ = standardize_macro_features(train, val, test)
        assert abs(t2["cpi_risk_norm"].mean()) < 0.1

    def test_no_cols_no_crash(self):
        train = pd.DataFrame({"a": [1.0, 2.0]})
        val = pd.DataFrame({"a": [3.0, 4.0]})
        test = pd.DataFrame({"a": [5.0, 6.0]})
        out = standardize_macro_features(train, val, test)
        assert len(out) == 3
