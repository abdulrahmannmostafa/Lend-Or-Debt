import os
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch
import src.scrape.fetch_fred_api as fred_module

# I use these helper functions to create consistent mock data for both CPI and GDP series


def _make_full_cpi_series():
    # Create a monthly series from Jan 2004 to Dec 2006 with a simple increasing pattern
    # freq="MS" means "Month Start", so we get the first day of each month
    dates = pd.date_range("2004-01-01", "2006-12-01", freq="MS")

    # Start at 100.0 and increase by 0.2 each month to create a simple pattern
    return pd.Series([100.0 + i * 0.2 for i in range(len(dates))], index=dates)


def _make_full_gdp_series():
    # Create a quarterly series from Jan 2004 to Oct 2006 with a simple increasing pattern
    # freq="QS" means "Quarter Start", so we get the first day of each quarter
    dates = pd.date_range("2004-01-01", "2006-10-01", freq="QS")

    # Start at 12000.0 and increase by 50 each quarter to create a simple pattern
    return pd.Series([12000.0 + i * 50 for i in range(len(dates))], index=dates)


# Marks this as reusable setup code that tests can request by name
@pytest.fixture()
def mock_fred(tmp_path):
    # tmp_path is a built-in pytest fixture that provides a temporary directory that is cleaned up after the test
    # here the tmp_path fixture is used to create a temporary directory for saving the CSV file during tests

    # Here I mock the Fred class from the fredapi module
    with patch("fredapi.Fred") as MockFred:

        # instance will be the mock object that simulates the behavior of the Fred class
        instance = MockFred.return_value

        # side_effect make get_series() return fake data depending on which series ID is passed — CPI series gets CPI data, anything else gets GDP data
        instance.get_series.side_effect = lambda series_id: (
            _make_full_cpi_series()
            if series_id == "CPIAUCSL"
            else _make_full_gdp_series()
        )

        # I yield not return because I don't want to still have the patch active after the test finishes — yielding allows the test to run with the mock in place
        # and then after the test completes, the patch will be automatically undone, restoring the original behavior of the Fred class for any other tests
        yield instance, tmp_path


class TestFetchMacroDataSchema:
    # I check that the function returns a DataFrame
    def test_returns_dataframe(self, mock_fred):
        instance, tmp = mock_fred
        from src.scrape.fetch_fred_api import fetch_macro_data

        result = fetch_macro_data(api_key="fake", save_path=str(tmp / "macro.csv"))
        assert isinstance(result, pd.DataFrame)

    # I check that the DataFrame has the expected columns
    def test_has_required_columns(self, mock_fred):
        instance, tmp = mock_fred
        from src.scrape.fetch_fred_api import fetch_macro_data

        result = fetch_macro_data(api_key="fake", save_path=str(tmp / "macro.csv"))
        assert {"Year", "Month", "CPI", "GDP"}.issubset(result.columns)

    # I check that there are no duplicate rows with the same Year and Month
    def test_no_duplicate_month_year_rows(self, mock_fred):
        instance, tmp = mock_fred
        from src.scrape.fetch_fred_api import fetch_macro_data

        result = fetch_macro_data(api_key="fake", save_path=str(tmp / "macro.csv"))
        assert not result.duplicated(subset=["Year", "Month"]).any()


class TestFetchMacroDataFiltering:
    # I check that the function correctly filters the data to only include rows from 2005, which is the expected behavior based on the mock data I set up
    def test_only_2005_rows(self, mock_fred):
        instance, tmp = mock_fred
        from src.scrape.fetch_fred_api import fetch_macro_data

        result = fetch_macro_data(api_key="fake", save_path=str(tmp / "macro.csv"))
        assert (result["Year"] == 2005).all()

    # I check that the Month column only contains valid month numbers (1-12) to ensure the data is correctly formatted and doesn't contain any invalid entries
    def test_months_in_valid_range(self, mock_fred):
        instance, tmp = mock_fred
        from src.scrape.fetch_fred_api import fetch_macro_data

        result = fetch_macro_data(api_key="fake", save_path=str(tmp / "macro.csv"))
        assert result["Month"].between(1, 12).all()

    # I check that there are no null values in the DataFrame to ensure data integrity
    def test_no_null_values(self, mock_fred):
        instance, tmp = mock_fred
        from src.scrape.fetch_fred_api import fetch_macro_data

        result = fetch_macro_data(api_key="fake", save_path=str(tmp / "macro.csv"))
        assert not result.isnull().any().any()


class TestFetchMacroDataPersistence:
    # I check that the function creates a CSV file at the specified path
    def test_csv_is_created(self, mock_fred):
        instance, tmp = mock_fred
        from src.scrape.fetch_fred_api import fetch_macro_data

        path = str(tmp / "macro.csv")
        fetch_macro_data(api_key="fake", save_path=path)
        assert os.path.exists(path)

    # I check that the content of the CSV file matches the DataFrame returned by the function to ensure that the data is being saved correctly and consistently
    def test_csv_content_matches_return(self, mock_fred):
        instance, tmp = mock_fred
        from src.scrape.fetch_fred_api import fetch_macro_data

        path = str(tmp / "macro.csv")

        result = fetch_macro_data(api_key="fake", save_path=path)

        on_disk = pd.read_csv(path)

        pd.testing.assert_frame_equal(
            result.reset_index(drop=True),
            on_disk.reset_index(drop=True),
            check_dtype=False,
        )

    def test_missing_api_key_raises(self):
        with patch("fredapi.Fred") as MockFred:
            MockFred.return_value.get_series.side_effect = Exception("Bad API key")
            from src.scrape.fetch_fred_api import fetch_macro_data

            with pytest.raises(Exception, match="Bad API key"):
                fetch_macro_data(api_key="bad", save_path="/tmp/x.csv")
