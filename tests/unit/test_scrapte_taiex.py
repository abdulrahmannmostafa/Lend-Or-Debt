import os
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch


# I create a helper function to generate a consistent mock DataFrame that simulates the daily stock data for the TAIEX index during the period of interest (April–September 2005)
def _make_daily_df():
    dates = pd.date_range("2005-04-01", "2005-09-30", freq="B")  # business days
    n = len(dates)
    df = pd.DataFrame(
        {
            "Date": dates,
            "Open": [6000.0 + i * 0.5 for i in range(n)],
            "High": [6050.0 + i * 0.5 for i in range(n)],
            "Low": [5950.0 + i * 0.5 for i in range(n)],
            "Close": [6010.0 + i * 0.5 for i in range(n)],
            "Volume": [1_000_000] * n,
        }
    )
    # yfinance returns a DatetimeIndex, not a column; simulate after reset_index
    return df


@pytest.fixture()
def mock_yfinance(tmp_path):
    # I mock the Ticker class from the yfinance module to return a fake response containing our generated daily DataFrame when the history() method is called
    with patch("src.scrape.scrape_taiex.yf.Ticker") as MockTicker:
        instance = MockTicker.return_value
        instance.history.return_value = _make_daily_df().set_index("Date")
        yield tmp_path


class TestScrapeTaiexSchema:
    # I check that the function returns a DataFrame to ensure that the output is in the expected format for downstream analysis
    def test_returns_dataframe(self, mock_yfinance):
        from src.scrape.scrape_taiex import scrape_taiex

        result = scrape_taiex(save_path=str(mock_yfinance / "taiex.csv"))
        assert isinstance(result, pd.DataFrame)

    # I check that the required columns are present in the DataFrame to ensure that the data is structured correctly for downstream analysis
    def test_required_columns_present(self, mock_yfinance):
        from src.scrape.scrape_taiex import scrape_taiex

        result = scrape_taiex(save_path=str(mock_yfinance / "taiex.csv"))
        for col in ["Year", "Month", "TAIEX_close"]:
            assert col in result.columns, f"Missing column: {col}"

    # I check that all aggregation columns (open, high, low, close, avg) are present in the DataFrame to ensure that the function is correctly calculating and including all necessary aggregated values for analysis
    def test_all_aggregation_columns_present(self, mock_yfinance):
        from src.scrape.scrape_taiex import scrape_taiex

        result = scrape_taiex(save_path=str(mock_yfinance / "taiex.csv"))
        for col in [
            "TAIEX_open",
            "TAIEX_high",
            "TAIEX_low",
            "TAIEX_close",
            "TAIEX_avg",
        ]:
            assert col in result.columns


class TestScrapeTaiexAggregation:
    # I check that the DataFrame contains exactly 6 rows, which corresponds to the 6 months of data (April–September) that we expect to extract from the daily data
    def test_exactly_6_monthly_rows(self, mock_yfinance):
        from src.scrape.scrape_taiex import scrape_taiex

        result = scrape_taiex(save_path=str(mock_yfinance / "taiex.csv"))
        assert len(result) == 6

    # I check that the Month column contains the correct month numbers (4-9) to ensure that the data is correctly filtered to include only the relevant months from the daily data
    def test_correct_months_present(self, mock_yfinance):
        from src.scrape.scrape_taiex import scrape_taiex

        result = scrape_taiex(save_path=str(mock_yfinance / "taiex.csv"))
        assert set(result["Month"]) == {4, 5, 6, 7, 8, 9}

    # I check that the Year column is 2005 for all rows to confirm that the data corresponds to the correct year as specified in the test setup
    def test_year_is_2005(self, mock_yfinance):
        from src.scrape.scrape_taiex import scrape_taiex

        result = scrape_taiex(save_path=str(mock_yfinance / "taiex.csv"))
        assert (result["Year"] == 2005).all()

    # I check that there are no duplicate rows with the same Year and Month to ensure that the aggregation process is correctly grouping the daily data into unique monthly entries without creating duplicates
    def test_no_duplicate_month_rows(self, mock_yfinance):
        from src.scrape.scrape_taiex import scrape_taiex

        result = scrape_taiex(save_path=str(mock_yfinance / "taiex.csv"))
        assert not result.duplicated(subset=["Year", "Month"]).any()

    # I check that the high values are greater than or equal to the low values to ensure that the aggregated data is consistent and that there are no errors in the calculation process
    def test_high_gte_low(self, mock_yfinance):
        from src.scrape.scrape_taiex import scrape_taiex

        result = scrape_taiex(save_path=str(mock_yfinance / "taiex.csv"))
        assert (result["TAIEX_high"] >= result["TAIEX_low"]).all()

    # I check that the close values are between the high and low values to ensure that the aggregated data is consistent and that there are no errors in the calculation process
    def test_close_between_high_and_low(self, mock_yfinance):
        from src.scrape.scrape_taiex import scrape_taiex

        result = scrape_taiex(save_path=str(mock_yfinance / "taiex.csv"))

        # I check that the close price is greater than or equal to the low price and less than or equal to the high price for each month to ensure that the aggregated data is consistent and that there are no errors in the calculation process
        assert (result["TAIEX_close"] >= result["TAIEX_low"]).all()
        assert (result["TAIEX_close"] <= result["TAIEX_high"]).all()


class TestScrapeTaiexEmptyResponse:
    def test_returns_empty_df_when_yfinance_empty(self, tmp_path):
        # I mock the Ticker class from the yfinance module to return an empty DataFrame when the history() method is called
        # and then check that the scrape_taiex function returns an empty DataFrame in this case to ensure that the function can handle cases where there is no data available without crashing
        with patch("src.scrape.scrape_taiex.yf.Ticker") as MockTicker:
            MockTicker.return_value.history.return_value = pd.DataFrame()
            from src.scrape.scrape_taiex import scrape_taiex

            result = scrape_taiex(save_path=str(tmp_path / "taiex.csv"))
            assert isinstance(result, pd.DataFrame)
            assert result.empty


class TestScrapeTaiexPersistence:
    # I check that the function creates a CSV file at the specified path to verify that the data is being saved correctly for future use
    def test_csv_is_created(self, mock_yfinance):
        from src.scrape.scrape_taiex import scrape_taiex

        path = str(mock_yfinance / "taiex.csv")
        scrape_taiex(save_path=path)
        assert os.path.exists(path)

    # I check that the content of the CSV file matches the DataFrame returned by the function to ensure that the data is being saved correctly and consistently
    def test_csv_matches_returned_dataframe(self, mock_yfinance):
        from src.scrape.scrape_taiex import scrape_taiex

        path = str(mock_yfinance / "taiex.csv")
        result = scrape_taiex(save_path=path)
        on_disk = pd.read_csv(path)

        # reset_index is used to ignore the index when comparing the DataFrames, and check_dtype=False allows for comparison even if the data types differ
        # because I have no control over the data types that will be inferred when reading from CSV
        pd.testing.assert_frame_equal(
            result.reset_index(drop=True),
            on_disk.reset_index(drop=True),
            check_dtype=False,
        )
