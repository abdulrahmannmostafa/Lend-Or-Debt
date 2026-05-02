import os
import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture()
def cbc_result(tmp_path):
    with patch("src.scrape.parse_cbc_pdf.requests.get") as mock_get:
        mock_get.side_effect = Exception("Network unavailable")
        from src.scrape.parse_cbc_pdf import parse_cbc_rates

        result = parse_cbc_rates(save_path=str(tmp_path / "cbc.csv"))
    return result, tmp_path


class TestParseCbcRatesSchema:
    def test_returns_dataframe(self, cbc_result):
        result, _ = cbc_result
        assert isinstance(result, pd.DataFrame)

    def test_required_columns_present(self, cbc_result):
        result, _ = cbc_result
        for col in ["Year", "Month", "Discount_Rate"]:
            assert col in result.columns

    def test_no_nulls(self, cbc_result):
        result, _ = cbc_result
        assert not result.isnull().any().any()


class TestParseCbcRatesBusinessLogic:
    def test_exactly_6_rows(self, cbc_result):
        """One row per month Apr-Sep 2005."""
        result, _ = cbc_result
        assert len(result) == 6

    def test_correct_months(self, cbc_result):
        result, _ = cbc_result
        assert set(result["Month"]) == {4, 5, 6, 7, 8, 9}

    def test_year_is_2005(self, cbc_result):
        result, _ = cbc_result
        assert (result["Year"] == 2005).all()

    def test_april_june_rate_is_2_000(self, cbc_result):
        result, _ = cbc_result
        pre_july = result[result["Month"].isin([4, 5, 6])]["Discount_Rate"]
        assert np.allclose(pre_july.to_numpy(), 2.0)

    def test_july_september_rate_is_2_125(self, cbc_result):
        result, _ = cbc_result
        post_july = result[result["Month"].isin([7, 8, 9])]["Discount_Rate"]
        assert np.allclose(post_july.to_numpy(), 2.125)

    def test_rate_increases_monotonically_across_hike(self, cbc_result):
        result, _ = cbc_result
        june_rate = result[result["Month"] == 6]["Discount_Rate"].iloc[0]
        july_rate = result[result["Month"] == 7]["Discount_Rate"].iloc[0]
        assert july_rate >= june_rate

    def test_rates_within_2005_cbc_bounds(self, cbc_result):
        result, _ = cbc_result
        assert result["Discount_Rate"].between(1.5, 2.5).all()


class TestParseCbcRatesPersistence:
    def test_csv_is_created(self, cbc_result):
        result, tmp = cbc_result
        assert os.path.exists(str(tmp / "cbc.csv"))

    def test_csv_matches_return(self, cbc_result):
        result, tmp = cbc_result
        on_disk = pd.read_csv(str(tmp / "cbc.csv"))
        pd.testing.assert_frame_equal(
            result[["Year", "Month", "Discount_Rate"]].reset_index(drop=True),
            on_disk[["Year", "Month", "Discount_Rate"]].reset_index(drop=True),
            check_dtype=False,
        )
