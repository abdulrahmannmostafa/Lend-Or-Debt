import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

from src.pipeline.data_validation import (
    validate_accuracy,
    validate_consistency,
    validate_completeness,
    validate_uniqueness,
    validate_outliers,
    validate_timeliness,
    validate_distribution,
    validate_relationships,
    validate_integrity,
    compute_quality_metrics,
)

TARGET_COL = "default payment next month"


def _valid_df(n=200, seed=7):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "LIMIT_BAL": rng.uniform(10_000, 800_000, n),
            "SEX": rng.integers(1, 3, n),
            "EDUCATION": rng.integers(1, 5, n),
            "MARRIAGE": rng.integers(1, 4, n),
            "AGE": rng.integers(22, 70, n),
            **{f"PAY_{k}": rng.integers(-2, 9, n) for k in [0, 2, 3, 4, 5, 6]},
            **{f"BILL_AMT{i}": rng.uniform(0, 80_000, n) for i in range(1, 7)},
            **{f"PAY_AMT{i}": rng.uniform(0, 30_000, n) for i in range(1, 7)},
            "avg_bill": rng.uniform(0, 60_000, n),
            "avg_payment": rng.uniform(0, 30_000, n),
            "max_delinquency": rng.integers(0, 8, n).astype(float),
            "total_delinquency": rng.integers(0, 20, n).astype(float),
            "avg_macro_CPI": np.full(n, 104.5),
            "avg_macro_GDP": np.full(n, 12500.0),
            "avg_macro_TAIEX": np.full(n, 6200.0),
            "avg_macro_rate": np.full(n, 2.0625),
            "avg_macro_unemp": np.full(n, 4.09),
            TARGET_COL: rng.integers(0, 2, n),
        }
    )
    return df


@pytest.fixture
def valid_df():
    return _valid_df(n=300)


class TestValidateAccuracy:
    def test_returns_list(self, valid_df):
        results = validate_accuracy(valid_df)
        assert isinstance(results, list)

    def test_all_checks_have_required_keys(self, valid_df):
        for item in validate_accuracy(valid_df):
            assert {"check", "success", "observed", "expected"} <= item.keys()

    def test_valid_ages_pass(self, valid_df):
        results = validate_accuracy(valid_df)
        age_check = next((r for r in results if "AGE" in r["check"]), None)
        assert age_check is not None
        assert age_check["success"] is True

    def test_invalid_age_fails(self):
        df = _valid_df()
        df.loc[0, "AGE"] = 5  # underage
        results = validate_accuracy(df)
        age_check = next((r for r in results if "AGE" in r["check"]), None)
        assert age_check["success"] is False

    def test_macro_within_range_passes(self, valid_df):
        results = validate_accuracy(valid_df)
        cpi_check = next((r for r in results if "CPI" in r["check"]), None)
        assert cpi_check["success"] is True


class TestValidateConsistency:
    def test_returns_list(self, valid_df):
        assert isinstance(validate_consistency(valid_df), list)

    def test_negative_avg_payment_fails(self):
        df = _valid_df()
        df.loc[0, "avg_payment"] = -100
        results = validate_consistency(df)
        pay_check = next((r for r in results if "avg_payment" in r["check"]), None)
        assert pay_check["success"] is False

    def test_max_delinquency_above_8_fails(self):
        df = _valid_df()
        df.loc[0, "max_delinquency"] = 9
        results = validate_consistency(df)
        check = next((r for r in results if "max_delinquency" in r["check"]), None)
        assert check["success"] is False


class TestValidateCompleteness:
    def test_returns_list(self, valid_df):
        assert isinstance(validate_completeness(valid_df), list)

    def test_null_column_fails(self):
        df = _valid_df(n=300)
        df.loc[0, "AGE"] = np.nan
        results = validate_completeness(df)
        null_check = next((r for r in results if "'AGE'" in r["check"]), None)
        assert null_check is not None
        assert null_check["success"] is False

    def test_row_count_check_present(self, valid_df):
        results = validate_completeness(valid_df)
        row_check = next((r for r in results if "Row count" in r["check"]), None)
        assert row_check is not None


class TestValidateUniqueness:
    def test_returns_list(self, valid_df):
        assert isinstance(validate_uniqueness(valid_df), list)

    def test_exact_duplicates_fail(self):
        df = _valid_df(n=100)
        duped = pd.concat([df, df.head(3)], ignore_index=True)
        results = validate_uniqueness(duped)
        dup_check = next(
            (r for r in results if "exact duplicate" in r["check"].lower()), None
        )
        assert dup_check["success"] is False

    def test_no_duplicates_passes(self, valid_df):
        results = validate_uniqueness(valid_df)
        dup_check = next(
            (r for r in results if "exact duplicate" in r["check"].lower()), None
        )
        assert dup_check["success"] is True


class TestValidateOutliers:
    def test_returns_list(self, valid_df):
        assert isinstance(validate_outliers(valid_df), list)

    def test_all_three_methods_present(self, valid_df):
        results = validate_outliers(valid_df)
        labels = [r["check"] for r in results]
        assert any("Z-Score" in l for l in labels)
        assert any("IQR" in l for l in labels)
        assert any("Isolation Forest" in l for l in labels)

    def test_quality_metric_check_present(self, valid_df):
        results = validate_outliers(valid_df)
        quality = next(
            (r for r in results if "quality metric" in r["check"].lower()), None
        )
        assert quality is not None

    def test_no_numeric_cols_graceful(self):
        df = pd.DataFrame({"cat": ["a", "b", "c"]})
        results = validate_outliers(df)
        assert len(results) >= 1


class TestValidateTimeliness:
    def test_returns_list(self, valid_df):
        assert isinstance(validate_timeliness(valid_df), list)

    def test_valid_macro_passes_ceilings(self, valid_df):
        results = validate_timeliness(valid_df)
        ceiling_checks = [r for r in results if "ceiling" in r["check"].lower()]
        for c in ceiling_checks:
            assert c["success"] is True

    def test_post_2010_taiex_fails(self):
        df = _valid_df()
        df["avg_macro_TAIEX"] = 11_000.0  # above 10,000 ceiling
        results = validate_timeliness(df)
        taiex_check = next((r for r in results if "TAIEX" in r["check"]), None)
        assert taiex_check["success"] is False

    def test_distinct_value_check_present(self, valid_df):
        results = validate_timeliness(valid_df)
        distinct = [r for r in results if "distinct values" in r["check"]]
        assert len(distinct) > 0


# ─── validate_distribution ───────────────────────────────────────────────────


class TestValidateDistribution:
    def test_returns_list(self, valid_df):
        assert isinstance(validate_distribution(valid_df), list)

    def test_default_rate_check_present(self, valid_df):
        results = validate_distribution(valid_df)
        dr_check = next((r for r in results if "Default rate" in r["check"]), None)
        assert dr_check is not None

    def test_ks_tests_run(self, valid_df):
        results = validate_distribution(valid_df)
        ks_checks = [r for r in results if "KS test" in r["check"]]
        assert len(ks_checks) > 0

    def test_distribution_profiles_present(self, valid_df):
        results = validate_distribution(valid_df)
        profiles = [r for r in results if "distribution profile" in r["check"].lower()]
        assert len(profiles) > 0


class TestValidateRelationships:
    def test_returns_list(self, valid_df):
        assert isinstance(validate_relationships(valid_df), list)

    def test_pearson_spearman_checks_present(self, valid_df):
        results = validate_relationships(valid_df)
        ps_checks = [r for r in results if "Pearson vs Spearman" in r["check"]]
        assert len(ps_checks) > 0

    def test_multicollinearity_check_present(self, valid_df):
        results = validate_relationships(valid_df)
        mc_check = next((r for r in results if "collinear" in r["check"].lower()), None)
        assert mc_check is not None

    def test_limit_bal_negative_correlation(self, valid_df):
        results = validate_relationships(valid_df)
        lb_check = next(
            (
                r
                for r in results
                if "LIMIT_BAL" in r["check"] and "negatively" in r["check"]
            ),
            None,
        )
        # LIMIT_BAL may or may not correlate negatively in a small synthetic df;
        # just confirm the check is run
        assert lb_check is not None


class TestValidateIntegrity:
    def test_returns_list(self, valid_df):
        assert isinstance(validate_integrity(valid_df), list)

    def test_missing_expected_col_fails(self):
        df = _valid_df()
        df = df.drop(columns=["LIMIT_BAL"])
        results = validate_integrity(df)
        col_check = next(
            (r for r in results if "expected columns" in r["check"].lower()), None
        )
        assert col_check["success"] is False

    def test_non_binary_target_fails(self):
        df = _valid_df()
        df.loc[0, TARGET_COL] = 5
        results = validate_integrity(df)
        binary_check = next(
            (r for r in results if "binary" in r["check"].lower()), None
        )
        assert binary_check["success"] is False

    def test_object_dtype_col_fails(self):
        df = _valid_df()
        df["LIMIT_BAL"] = df["LIMIT_BAL"].astype(str)
        results = validate_integrity(df)
        dtype_check = next(
            (r for r in results if "object dtype" in r["check"].lower()), None
        )
        assert dtype_check["success"] is False


class TestComputeQualityMetrics:
    def _build_results(self, df):
        return {
            "1_Accuracy": validate_accuracy(df),
            "2_Consistency": validate_consistency(df),
            "3_Completeness": validate_completeness(df),
            "4_Uniqueness": validate_uniqueness(df),
            "5_Outliers": validate_outliers(df),
            "6_Timeliness": validate_timeliness(df),
            "7_Distribution": validate_distribution(df),
            "8_Relationships": validate_relationships(df),
        }

    def test_returns_dict(self, valid_df):
        results = self._build_results(valid_df)
        metrics = compute_quality_metrics(valid_df, results)
        assert isinstance(metrics, dict)

    def test_completeness_is_100_for_clean_df(self, valid_df):
        results = self._build_results(valid_df)
        metrics = compute_quality_metrics(valid_df, results)
        assert metrics["Completeness"]["value"] == 100.0

    def test_uniqueness_is_100_for_no_dups(self, valid_df):
        results = self._build_results(valid_df)
        metrics = compute_quality_metrics(valid_df, results)
        assert metrics["Uniqueness"]["value"] == 100.0

    def test_all_expected_dimensions_present(self, valid_df):
        results = self._build_results(valid_df)
        metrics = compute_quality_metrics(valid_df, results)
        for key in [
            "Completeness",
            "Uniqueness",
            "Accuracy",
            "Consistency",
            "Outliers",
            "Timeliness",
            "Distribution",
            "Relationships",
        ]:
            assert key in metrics
