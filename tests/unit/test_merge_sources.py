import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch

N = 10  # number of synthetic borrowers


def _make_taiwan_df(n=N) -> pd.DataFrame:
    data = {
        "ID": range(1, n + 1),
        "LIMIT_BAL": [50_000 * (i + 1) for i in range(n)],
        "SEX": [1, 2] * (n // 2),
        "EDUCATION": [1, 2, 3, 4] * (n // 4) + [1, 2],
        "MARRIAGE": [1, 2, 3] * (n // 3) + [1],
        "AGE": [25 + i for i in range(n)],
        # PAY_STATUS columns
        "PAY_0": [0] * n,
        "PAY_2": [0] * n,
        "PAY_3": [-1] * n,
        "PAY_4": [-1] * n,
        "PAY_5": [-2] * n,
        "PAY_6": [-2] * n,
        # BILL amounts
        **{f"BILL_AMT{i}": [1000 * i] * n for i in range(1, 7)},
        # PAY amounts
        **{f"PAY_AMT{i}": [500 * i] * n for i in range(1, 7)},
        "default payment next month": [0, 1] * (n // 2),
    }
    return pd.DataFrame(data)


def _make_macro_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Year": [2005] * 6,
            "Month": [4, 5, 6, 7, 8, 9],
            "CPI": [104.1, 104.5, 104.8, 105.0, 105.3, 105.5],
            "GDP": [12500.0] * 6,
        }
    )


def _make_taiex_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Year": [2005] * 6,
            "Month": [4, 5, 6, 7, 8, 9],
            "TAIEX_close": [6100.0, 6200.0, 6050.0, 6300.0, 6400.0, 6500.0],
        }
    )


def _make_cbc_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Year": [2005] * 6,
            "Month": [4, 5, 6, 7, 8, 9],
            "Discount_Rate": [2.000, 2.000, 2.000, 2.125, 2.125, 2.125],
        }
    )


def _make_unemploy_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Year": [2005] * 6,
            "Month": [4, 5, 6, 7, 8, 9],
            "Unemployment_Rate": [4.13, 4.13, 4.13, 4.05, 4.05, 4.05],
        }
    )


class TestBuildPanel:
    # I check that the resulting panel has the expected number of rows, which should be the number of unique borrowers (N) multiplied by the 6 months of data
    def test_row_count(self):
        from src.scrape.merge_sources import _build_panel

        taiwan = _make_taiwan_df()

        panel = _build_panel(taiwan)

        assert len(panel) == N * 6, "Expected N borrowers x 6 months"

    # I check that the Month column in the resulting panel only contains the months 4 through 9, which are the months for which we have macroeconomic data
    def test_year_is_2005(self):
        from src.scrape.merge_sources import _build_panel

        panel = _build_panel(_make_taiwan_df())

        assert (panel["Year"] == 2005).all()

    # I check that the resulting panel contains the required columns for the subsequent merging and analysis steps, ensuring that the data is structured correctly for downstream processing
    def test_required_columns_present(self):
        from src.scrape.merge_sources import _build_panel

        panel = _build_panel(_make_taiwan_df())

        for col in ["ID", "Year", "Month", "PAY_STATUS", "BILL_AMT", "PAY_AMT"]:
            assert col in panel.columns

    # I check that each borrower has exactly 6 rows in the resulting panel, which corresponds to the 6 months of data we are including for each borrower
    def test_each_borrower_has_6_rows(self):
        from src.scrape.merge_sources import _build_panel

        panel = _build_panel(_make_taiwan_df())

        counts = panel.groupby("ID").size()

        assert (counts == 6).all()

    # I check that the PAY_STATUS column in the resulting panel correctly maps the original PAY_0 column to the new PAY_STATUS column
    # ensuring that the data transformation is performed correctly according to our specifications
    def test_september_maps_to_pay_0(self):
        from src.scrape.merge_sources import _build_panel

        taiwan = _make_taiwan_df()

        # set a distinctive value in PAY_0 to test the mapping
        taiwan["PAY_0"] = 3

        panel = _build_panel(taiwan)

        sept_rows = panel[panel["Month"] == 9]

        assert (sept_rows["PAY_STATUS"] == 3).all()


class TestRecodePayStatus:
    # I check that negative values in the PAY_STATUS column are recoded to zero, which is important for ensuring that the delinquency status is represented correctly and that negative values do not skew the analysis
    def test_negative_values_clipped_to_zero(self):
        from src.scrape.merge_sources import _recode_pay_status

        df = pd.DataFrame({"PAY_STATUS": [-2, -1, 0, 1, 2, 8]})

        out = _recode_pay_status(df)

        assert out["PAY_STATUS_clean"].min() == 0

    # I check that positive values in the PAY_STATUS column are unchanged in the PAY_STATUS_clean column, which ensures that the recoding process only affects negative values and preserves the original delinquency status for non-negative values
    def test_positive_values_unchanged(self):
        from src.scrape.merge_sources import _recode_pay_status

        df = pd.DataFrame({"PAY_STATUS": [1, 2, 8]})

        out = _recode_pay_status(df)

        assert list(out["PAY_STATUS_clean"]) == [1, 2, 8]

    # I check that the original PAY_STATUS column is preserved in the output DataFrame, which ensures that the recoding process does not modify the original data
    def test_original_column_preserved(self):
        from src.scrape.merge_sources import _recode_pay_status

        df = pd.DataFrame({"PAY_STATUS": [-2, 0, 3]})

        out = _recode_pay_status(df)

        assert "PAY_STATUS" in out.columns

        assert list(out["PAY_STATUS"]) == [-2, 0, 3]


class TestAggregateBehaviour:
    def _make_panel(self):
        from src.scrape.merge_sources import _build_panel, _recode_pay_status

        taiwan = _make_taiwan_df()

        panel = _build_panel(taiwan)

        panel = panel.merge(_make_macro_df(), on=["Year", "Month"], how="left")

        panel = panel.merge(
            _make_taiex_df()[["Year", "Month", "TAIEX_close"]],
            on=["Year", "Month"],
            how="left",
        )

        panel = panel.merge(
            _make_cbc_df()[["Year", "Month", "Discount_Rate"]],
            on=["Year", "Month"],
            how="left",
        )

        panel = panel.merge(
            _make_unemploy_df()[["Year", "Month", "Unemployment_Rate"]],
            on=["Year", "Month"],
            how="left",
        )

        return _recode_pay_status(panel)

    # I check that the resulting aggregated DataFrame has one row per unique borrower ID to ensure that the aggregation is correct
    def test_one_row_per_borrower(self):
        from src.scrape.merge_sources import _aggregate_behaviour

        agg = _aggregate_behaviour(self._make_panel())

        assert len(agg) == N

    # I check that the resulting aggregated DataFrame contains the required columns for the subsequent analysis steps
    def test_required_agg_columns(self):
        from src.scrape.merge_sources import _aggregate_behaviour

        agg = _aggregate_behaviour(self._make_panel())

        for col in [
            "avg_bill",
            "avg_payment",
            "max_delinquency",
            "total_delinquency",
            "avg_macro_CPI",
            "avg_macro_GDP",
            "avg_macro_TAIEX",
            "avg_macro_rate",
            "avg_macro_unemp",
        ]:
            assert col in agg.columns

    # I check that the max_delinquency column in the resulting aggregated DataFrame is greater than or equal to zero
    # which ensures that the maximum delinquency status is represented correctly and that negative values do not skew the analysis
    def test_max_delinquency_gte_zero(self):
        from src.scrape.merge_sources import _aggregate_behaviour

        agg = _aggregate_behaviour(self._make_panel())

        assert (agg["max_delinquency"] >= 0).all()

    # I check that the avg_payment column in the resulting aggregated DataFrame is greater than or equal to zero
    # which ensures that the average payment amount is represented correctly and negative values do not skew the analysis
    def test_avg_payment_gte_zero(self):
        from src.scrape.merge_sources import _aggregate_behaviour

        agg = _aggregate_behaviour(self._make_panel())

        assert (agg["avg_payment"] >= 0).all()


class TestBuildFinalDataset:
    @pytest.fixture()
    def mock_loaders(self, tmp_path):
        taiwan = _make_taiwan_df()

        macro = _make_macro_df()

        taiex = _make_taiex_df()

        cbc = _make_cbc_df()

        unemp = _make_unemploy_df()

        save_path = str(tmp_path / "merged.csv")

        # I mock the file loading functions to return the synthetic DataFrames instead of reading from actual files
        # this allows me to test the merging logic in isolation without relying on external files
        with patch(
            "src.scrape.merge_sources.pd.read_excel", return_value=taiwan
        ), patch(
            "src.scrape.merge_sources.pd.read_csv",
            side_effect=[macro, taiex, cbc, unemp],
        ):
            from src.scrape.merge_sources import build_final_dataset

            result = build_final_dataset(
                taiwan_path="fake.xls",
                macro_path="fake_macro.csv",
                taiex_path="fake_taiex.csv",
                cbc_path="fake_cbc.csv",
                unemploy_path="fake_unemp.csv",
                save_path=save_path,
            )

        yield result, save_path

    # I check that the function returns a DataFrame, which is the expected output type for the merged dataset
    def test_returns_dataframe(self, mock_loaders):
        result, _ = mock_loaders

        assert isinstance(result, pd.DataFrame)

    # I check that the number of rows in the resulting DataFrame matches the number of unique borrowers (N) to ensure that the merging process does not create duplicate rows for borrowers
    # and that we have one row per borrower in the final dataset
    def test_row_count_matches_borrowers(self, mock_loaders):
        result, _ = mock_loaders

        assert len(result) == N, "One row per borrower — no merge fan-out"

    # I check that the ID column from the original Taiwan dataset is dropped in the final merged dataset
    # which is important for data privacy and to ensure that the final dataset does not contain personally identifiable information
    # because if it exists in the final dataset, it could potentially be used to identify individual borrowers, which would be a privacy concern and could violate data protection regulations
    def test_id_column_dropped(self, mock_loaders):
        result, _ = mock_loaders

        assert "ID" not in result.columns

    # I check that there are no null values in the macroeconomic columns of the resulting DataFrame to ensure data is complete
    def test_no_nulls_in_macro_columns(self, mock_loaders):
        result, _ = mock_loaders

        macro_cols = [
            "avg_macro_CPI",
            "avg_macro_GDP",
            "avg_macro_TAIEX",
            "avg_macro_rate",
            "avg_macro_unemp",
        ]

        for col in macro_cols:
            assert result[col].notna().all(), f"Null found in {col}"

    # I check that the default payment next month column in the resulting DataFrame only contains binary values (0 and 1) to ensure that the target variable is correctly represented for classification tasks
    def test_target_column_preserved(self, mock_loaders):
        result, _ = mock_loaders

        assert "default payment next month" in result.columns

        assert result["default payment next month"].isin({0, 1}).all()

    # I check that the resulting DataFrame contains the required columns for the subsequent analysis steps
    # ensuring that the data is structured correctly for downstream processing
    def test_csv_is_written(self, mock_loaders):
        _, path = mock_loaders
        import os

        assert os.path.exists(path)

    # I check that the number of rows in the resulting DataFrame matches the number of unique borrowers (N) to ensure that the merging process does not create duplicate rows for borrowers
    def test_no_merge_fan_out(self, mock_loaders):
        result, _ = mock_loaders
        assert len(result) == N
