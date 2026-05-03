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
    # I check that the function returns a list to ensure that the output is in the expected format for downstream processing and reporting of validation results
    def test_returns_list(self, valid_df):
        results = validate_accuracy(valid_df)
        assert isinstance(results, list)

    # I check that each item in the results list contains the required keys ('check', 'success', 'observed', 'expected') to ensure that the validation results are structured correctly for interpretation and reporting
    def test_all_checks_have_required_keys(self, valid_df):
        for item in validate_accuracy(valid_df):
            assert {"check", "success", "observed", "expected"} <= item.keys()

    # I check that the AGE column is validated correctly by confirming that valid ages pass the check and invalid ages fail the check to ensure that the function is correctly assessing the accuracy of the AGE data according to expected criteria (like, age should be between 18 and 100)
    def test_valid_ages_pass(self, valid_df):
        results = validate_accuracy(valid_df)
        age_check = next((r for r in results if "AGE" in r["check"]), None)

        assert age_check is not None
        assert age_check["success"] is True

    # I check that macroeconomic indicators (CPI, GDP, TAIEX, rate, unemp) are validated against expected ranges
    # and that values within those ranges pass the checks while values outside those ranges fail the checks to ensure that the function is correctly assessing the accuracy
    # of the macroeconomic data according to expected criteria based on historical values and economic knowledge
    def test_invalid_age_fails(self):
        df = _valid_df()
        df.loc[0, "AGE"] = 5  # underage
        results = validate_accuracy(df)

        # I use next() with a generator expression to find the first check in the results that contains "AGE" in its "check" description
        # and then confirm that this check failed successfully to verify that the function is correctly validating the
        # AGE values according to expected criteria (like, age should be between 18 and 100)
        age_check = next((r for r in results if "AGE" in r["check"]), None)

        assert age_check["success"] is False

    # I check that the macroeconomic indicators are validated correctly by confirming that values within expected ranges pass the checks and values outside those ranges fail the checks
    # to ensure that the function is correctly assessing the accuracy of the macroeconomic data according to expected criteria
    #  based on historical values and economic knowledge
    def test_macro_within_range_passes(self, valid_df):
        results = validate_accuracy(valid_df)

        # I use next() with a generator expression to find the first check in the results that contains "CPI" in its "check" description
        # and then confirm that this check passed successfully to verify that the function is correctly validating the CPI values according to expected criteria
        cpi_check = next((r for r in results if "CPI" in r["check"]), None)

        assert cpi_check["success"] is True


class TestValidateConsistency:
    # I check that the function returns a list to ensure that the output is in the expected format for downstream processing and reporting of validation results
    def test_returns_list(self, valid_df):
        assert isinstance(validate_consistency(valid_df), list)

    # I check that the function correctly identifies negative average payments as inconsistent with expected financial behavior and flags
    # them as failures in the consistency check to ensure that the function is correctly assessing the consistency of the
    # avg_payment data according to expected financial norms (like, average payment should not be negative)
    def test_negative_avg_payment_fails(self):
        df = _valid_df()

        # get the first row of the DataFrame and set the avg_payment value to -100 to create an inconsistent scenario where the average payment is negative, which is not expected in financial data
        df.loc[0, "avg_payment"] = -100

        results = validate_consistency(df)
        pay_check = next((r for r in results if "avg_payment" in r["check"]), None)

        assert pay_check["success"] is False

    # I check that the function correctly identifies cases where the maximum delinquency is greater than the total delinquency as inconsistent with expected financial behavior and flags
    def test_max_delinquency_above_8_fails(self):
        df = _valid_df()

        # get the first row of the DataFrame and set the max_delinquency value to 9 to create an inconsistent scenario where the maximum delinquency exceeds the total delinquency, which is not expected in financial data
        df.loc[0, "max_delinquency"] = 9

        results = validate_consistency(df)
        check = next((r for r in results if "max_delinquency" in r["check"]), None)

        assert check["success"] is False


class TestValidateCompleteness:
    # I check that the function returns a list to ensure that the output is in the expected format for downstream processing and reporting of validation results
    def test_returns_list(self, valid_df):
        assert isinstance(validate_completeness(valid_df), list)

    # I check that the function correctly identifies null values in the AGE column and flags them as failures in the
    # completeness check to ensure that the function is correctly
    # assessing the completeness of the AGE data according to expected criteria (like, age should not be null)
    def test_null_column_fails(self):
        df = _valid_df(n=300)

        # get the first row of the DataFrame and set the AGE value to NaN to create an incomplete scenario where the age is missing, which is not expected in the data
        df.loc[0, "AGE"] = np.nan

        results = validate_completeness(df)

        null_check = next((r for r in results if "'AGE'" in r["check"]), None)

        assert null_check is not None
        assert null_check["success"] is False

    # I check that the function includes a check for row count to ensure that the completeness validation is
    #  assessing whether the number of rows in the DataFrame meets expected criteria (like, not being zero or below a certain threshold) as part of its evaluation of data completeness
    def test_row_count_check_present(self, valid_df):
        results = validate_completeness(valid_df)

        # I use next() with a generator expression to find the first check in the results that contains "Row count" in its "check" description to confirm
        # that the function is including a check for row count as part of its completeness validation process
        row_check = next((r for r in results if "Row count" in r["check"]), None)
        assert row_check is not None


class TestValidateUniqueness:
    # I check that the function returns a list to ensure that the output is in the expected format for downstream processing and reporting of validation results
    def test_returns_list(self, valid_df):
        assert isinstance(validate_uniqueness(valid_df), list)

    # I check that the function correctly identifies exact duplicates and flags them as failures in the uniqueness check
    def test_exact_duplicates_fail(self):
        df = _valid_df(n=100)

        # I create a new DataFrame called duped by concatenating the original DataFrame with a subset of itself (the first 3 rows)
        # to create exact duplicate rows in the resulting DataFrame, which should trigger a failure in the uniqueness
        # check when passed to the validate_uniqueness function
        duped = pd.concat([df, df.head(3)], ignore_index=True)

        results = validate_uniqueness(duped)

        # I use next() with a generator expression to find the first check in the results that contains "exact duplicate" in its "check" description
        # and then confirm that this check failed successfully to verify that the function is correctly
        #  identifying exact duplicates and flagging them as failures in the uniqueness check
        dup_check = next(
            (r for r in results if "exact duplicate" in r["check"].lower()), None
        )

        assert dup_check["success"] is False

    # I check that the function correctly identifies when there are no exact duplicates and
    # flags this as a success in the uniqueness check to ensure that the function is correctly
    # assessing the uniqueness of the data according to expected criteria (like, not having exact duplicate rows)
    def test_no_duplicates_passes(self, valid_df):
        results = validate_uniqueness(valid_df)

        # I use next() with a generator expression to find the first check in the results that contains "exact duplicate" in its "check" description
        # and then confirm that this check passed successfully to verify that the function is correctly identifying when
        # there are no exact duplicates and flagging this as a success in the uniqueness check
        dup_check = next(
            (r for r in results if "exact duplicate" in r["check"].lower()), None
        )

        assert dup_check["success"] is True


class TestValidateOutliers:
    # I check that the function returns a list to ensure that the output is in the expected format for downstream processing and reporting of validation results
    def test_returns_list(self, valid_df):
        assert isinstance(validate_outliers(valid_df), list)

    # I check that the function includes checks for multiple outlier detection methods (Z-Score, IQR, Isolation Forest)
    # to ensure that the outlier validation is comprehensive and assesses outliers using different techniques for robustness
    def test_all_three_methods_present(self, valid_df):
        results = validate_outliers(valid_df)
        labels = [r["check"] for r in results]

        assert any("Z-Score" in l for l in labels)
        assert any("IQR" in l for l in labels)
        assert any("Isolation Forest" in l for l in labels)

    # I check that the function includes a check for quality metrics to ensure that
    # the outlier validation is assessing the impact of outliers on overall data quality as part of its evaluation process
    def test_quality_metric_check_present(self, valid_df):
        results = validate_outliers(valid_df)

        # I use next() with a generator expression to find the first check in the results that contains "quality metric" in its "check" description to confirm
        # that the function is including a check for quality metrics as part of its outlier validation
        quality = next(
            (r for r in results if "quality metric" in r["check"].lower()), None
        )

        assert quality is not None

    # I check that the function can handle cases where there are no numeric columns to assess for outliers without crashing
    # and returns appropriate results to ensure that the function is robust and can gracefully handle edge cases in the data
    def test_no_numeric_cols_graceful(self):
        df = pd.DataFrame({"cat": ["a", "b", "c"]})
        results = validate_outliers(df)

        assert len(results) >= 1


class TestValidateTimeliness:
    # I check that the function returns a list to ensure that the output is in the expected format for downstream processing and reporting of validation results
    def test_returns_list(self, valid_df):
        assert isinstance(validate_timeliness(valid_df), list)

    # I check that the function includes checks for macroeconomic indicator ceilings to ensure that the timeliness validation is
    #  assessing whether the macroeconomic data falls within expected historical ranges as part of its evaluation process
    def test_valid_macro_passes_ceilings(self, valid_df):
        results = validate_timeliness(valid_df)

        ceiling_checks = [r for r in results if "ceiling" in r["check"].lower()]

        for c in ceiling_checks:
            assert c["success"] is True

    # I check that the function correctly identifies when macroeconomic indicators exceed expected ceilings
    # and flags them as failures in the timeliness check to ensure that the function is correctly assessing
    # the timeliness of the macroeconomic data according to expected historical ranges
    def test_post_2010_taiex_fails(self):
        df = _valid_df()

        df["avg_macro_TAIEX"] = 11_000.0  # above 10,000 ceiling

        results = validate_timeliness(df)

        taiex_check = next((r for r in results if "TAIEX" in r["check"]), None)

        assert taiex_check["success"] is False

    # I check that the function includes a check for distinct values in the Month column
    # to ensure that the timeliness validation is assessing whether the data includes a
    #  reasonable distribution of months as part of its evaluation process
    def test_distinct_value_check_present(self, valid_df):
        results = validate_timeliness(valid_df)
        distinct = [r for r in results if "distinct values" in r["check"]]
        assert len(distinct) > 0


class TestValidateDistribution:
    # I check that the function returns a list to ensure that the output is
    # in the expected format for downstream processing and reporting of validation results
    def test_returns_list(self, valid_df):
        assert isinstance(validate_distribution(valid_df), list)

    # I check that the function includes a check for default rate distribution to ensure that the distribution validation
    # is assessing whether the target variable's distribution aligns with expected patterns as part of its evaluation process
    def test_default_rate_check_present(self, valid_df):
        results = validate_distribution(valid_df)
        dr_check = next((r for r in results if "Default rate" in r["check"]), None)
        assert dr_check is not None

    # I check that the function includes checks for distribution profiles of numeric columns to ensure that the
    # distribution validation is assessing whether the numeric features have distributions that align with expected patterns as part of its evaluation process
    def test_ks_tests_run(self, valid_df):
        results = validate_distribution(valid_df)
        ks_checks = [r for r in results if "KS test" in r["check"]]
        assert len(ks_checks) > 0

    # I check that the function includes checks for distribution profiles of numeric columns to
    # ensure that the distribution validation is assessing whether the numeric features have
    # distributions that align with expected patterns as part of its evaluation process
    def test_distribution_profiles_present(self, valid_df):
        results = validate_distribution(valid_df)
        profiles = [r for r in results if "distribution profile" in r["check"].lower()]
        assert len(profiles) > 0


class TestValidateRelationships:
    # I check that the function returns a list to ensure that the output is in the expected format for downstream processing and reporting of validation results
    def test_returns_list(self, valid_df):
        assert isinstance(validate_relationships(valid_df), list)

    # I check that the function includes checks for Pearson vs Spearman correlation
    def test_pearson_spearman_checks_present(self, valid_df):
        results = validate_relationships(valid_df)
        ps_checks = [r for r in results if "Pearson vs Spearman" in r["check"]]
        assert len(ps_checks) > 0

    # I check that the function includes a check for multicollinearity to ensure that the relationship validation
    # is assessing whether the features have high linear correlations as part of its evaluation process
    def test_multicollinearity_check_present(self, valid_df):
        results = validate_relationships(valid_df)
        mc_check = next((r for r in results if "collinear" in r["check"].lower()), None)
        assert mc_check is not None

    # I check that the function includes a check for negative correlation between LIMIT_BAL and the target variable to ensure that
    #  the relationship validation is assessing whether there is an expected negative correlation between credit limit
    #  and default risk as part of its evaluation process
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
    # I check that the function returns a list to ensure that the output is in the expected format for downstream processing and reporting of validation results
    def test_returns_list(self, valid_df):
        assert isinstance(validate_integrity(valid_df), list)

    # I check that the function includes a check for expected columns to ensure that the integrity validation is assessing whether
    # all required columns are present in the DataFrame as part of its evaluation process
    def test_missing_expected_col_fails(self):
        df = _valid_df()
        df = df.drop(columns=["LIMIT_BAL"])
        results = validate_integrity(df)
        col_check = next(
            (r for r in results if "expected columns" in r["check"].lower()), None
        )

        assert col_check["success"] is False

    # I check that the function includes a check for binary target variable to ensure that the integrity validation
    # is assessing whether the target variable is binary as part of its evaluation process
    def test_non_binary_target_fails(self):
        df = _valid_df()
        df.loc[0, TARGET_COL] = 5
        results = validate_integrity(df)
        binary_check = next(
            (r for r in results if "binary" in r["check"].lower()), None
        )

        assert binary_check["success"] is False

    # I check that the function includes a check for object dtype columns and correctly identifies them as failures to ensure that the
    #  integrity validation is assessing whether all columns have appropriate data types as part of its evaluation process
    def test_object_dtype_col_fails(self):
        df = _valid_df()
        df["LIMIT_BAL"] = df["LIMIT_BAL"].astype(str)
        results = validate_integrity(df)
        dtype_check = next(
            (r for r in results if "object dtype" in r["check"].lower()), None
        )
        assert dtype_check["success"] is False


class TestComputeQualityMetrics:
    # I define a helper method _build_results to run all the individual validation checks and compile their results into a dictionary
    # that can be passed to the compute_quality_metrics function for testing
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
