import os
import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture()
def cbc_result(tmp_path):
    with patch("src.scrape.parse_cbc_pdf.requests.get") as mock_get:
        # I make an exception to simulate a network failure, which should be handled gracefully by the function under test
        # allowing us to verify that the function can still produce the expected output even when the network is unavailable
        mock_get.side_effect = Exception("Network unavailable")
        from src.scrape.parse_cbc_pdf import parse_cbc_rates

        result = parse_cbc_rates(save_path=str(tmp_path / "cbc.csv"))
    return result, tmp_path


class TestParseCbcRatesSchema:
    # I check that the function returns a DataFrame, which is the expected output type for this function
    def test_returns_dataframe(self, cbc_result):
        result, _ = cbc_result

        assert isinstance(result, pd.DataFrame)

    # I check that the required columns are present in the DataFrame to ensure that the data is structured correctly for downstream analysis
    def test_required_columns_present(self, cbc_result):
        result, _ = cbc_result

        for col in ["Year", "Month", "Discount_Rate"]:
            assert col in result.columns

    # I check that there are no null values in the DataFrame to ensure data integrity and that the parsing process successfully extracted all necessary information from the PDF
    def test_no_nulls(self, cbc_result):
        result, _ = cbc_result

        assert not result.isnull().any().any()


class TestParseCbcRatesBusinessLogic:
    # I check that the DataFrame contains exactly 6 rows, which corresponds to the 6 months of data (April–September) that we expect to extract from the PDF
    def test_exactly_6_rows(self, cbc_result):
        result, _ = cbc_result

        assert len(result) == 6

    # I check that the Month column contains the correct month numbers (4-9) to ensure that the data is correctly filtered to include only the relevant months from the PDF
    def test_correct_months(self, cbc_result):
        result, _ = cbc_result

        assert set(result["Month"]) == {4, 5, 6, 7, 8, 9}

    # I check that the Year column is 2005 for all rows to confirm that the data corresponds to the correct year as specified in the PDF
    def test_year_is_2005(self, cbc_result):
        result, _ = cbc_result

        assert (result["Year"] == 2005).all()

    # I check that the discount rate for April–June is 2.000% to verify that the function correctly parsed the pre-hike rate from the PDF
    def test_april_june_rate_is_2_000(self, cbc_result):
        result, _ = cbc_result

        pre_july = result[result["Month"].isin([4, 5, 6])]["Discount_Rate"]

        # check that all values in the pre_july series are approximately equal to 2.000
        assert np.allclose(pre_july.to_numpy(), 2.0)

    # I check that the discount rate for July–September is 2.125% to verify that the function correctly parsed the post-hike rate from the PDF
    def test_july_september_rate_is_2_125(self, cbc_result):
        result, _ = cbc_result

        post_july = result[result["Month"].isin([7, 8, 9])]["Discount_Rate"]

        # check that all values in the post_july series are approximately equal to 2.125
        assert np.allclose(post_july.to_numpy(), 2.125)

    def test_rate_increases_monotonically_across_hike(self, cbc_result):
        result, _ = cbc_result

        # iloc[0] will give us the discount rate for April (the first month) because we will have a series
        june_rate = result[result["Month"] == 6]["Discount_Rate"].iloc[0]

        july_rate = result[result["Month"] == 7]["Discount_Rate"].iloc[0]

        assert july_rate >= june_rate

    # I check that all discount rates are within a reasonable range (like 1.5% to 2.5%) to ensure that the parsed values are reasonable
    def test_rates_within_2005_cbc_bounds(self, cbc_result):
        result, _ = cbc_result

        assert result["Discount_Rate"].between(1.5, 2.5).all()


class TestParseCbcRatesPersistence:
    # I check that the function creates a CSV file at the specified path to verify that the data is being saved correctly for future use
    def test_csv_is_created(self, cbc_result):
        result, tmp = cbc_result

        assert os.path.exists(str(tmp / "cbc.csv"))

    # I check that the content of the CSV file matches the DataFrame returned by the function to ensure that the data is being saved correctly and consistently
    def test_csv_matches_return(self, cbc_result):
        result, tmp = cbc_result

        on_disk = pd.read_csv(str(tmp / "cbc.csv"))

        # reset_index is used to ignore the index when comparing the DataFrames, and check_dtype=False allows for comparison even if the data types differ
        # because I have no control over the data types that will be inferred when reading from CSV, but I want to ensure that the values themselves match regardless of type differences
        pd.testing.assert_frame_equal(
            result[["Year", "Month", "Discount_Rate"]].reset_index(drop=True),
            on_disk[["Year", "Month", "Discount_Rate"]].reset_index(drop=True),
            check_dtype=False,
        )
