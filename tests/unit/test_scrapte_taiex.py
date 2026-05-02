import os
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch


def _make_daily_df():
    """Simulate 6 months of daily TAIEX data (Apr–Sep 2005)."""
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
    with patch("src.scrape.scrape_taiex.yf.Ticker") as MockTicker:
        instance = MockTicker.return_value
        instance.history.return_value = _make_daily_df().set_index("Date")
        yield tmp_path


class TestScrapeTaiexSchema:
    def test_returns_dataframe(self, mock_yfinance):
        from src.scrape.scrape_taiex import scrape_taiex

        result = scrape_taiex(save_path=str(mock_yfinance / "taiex.csv"))
        assert isinstance(result, pd.DataFrame)

    def test_required_columns_present(self, mock_yfinance):
        from src.scrape.scrape_taiex import scrape_taiex

        result = scrape_taiex(save_path=str(mock_yfinance / "taiex.csv"))
        for col in ["Year", "Month", "TAIEX_close"]:
            assert col in result.columns, f"Missing column: {col}"

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
    def test_exactly_6_monthly_rows(self, mock_yfinance):
        """Apr-Sep 2005 = 6 months."""
        from src.scrape.scrape_taiex import scrape_taiex

        result = scrape_taiex(save_path=str(mock_yfinance / "taiex.csv"))
        assert len(result) == 6

    def test_correct_months_present(self, mock_yfinance):
        from src.scrape.scrape_taiex import scrape_taiex

        result = scrape_taiex(save_path=str(mock_yfinance / "taiex.csv"))
        assert set(result["Month"]) == {4, 5, 6, 7, 8, 9}

    def test_year_is_2005(self, mock_yfinance):
        from src.scrape.scrape_taiex import scrape_taiex

        result = scrape_taiex(save_path=str(mock_yfinance / "taiex.csv"))
        assert (result["Year"] == 2005).all()

    def test_no_duplicate_month_rows(self, mock_yfinance):
        from src.scrape.scrape_taiex import scrape_taiex

        result = scrape_taiex(save_path=str(mock_yfinance / "taiex.csv"))
        assert not result.duplicated(subset=["Year", "Month"]).any()

    def test_high_gte_low(self, mock_yfinance):
        from src.scrape.scrape_taiex import scrape_taiex

        result = scrape_taiex(save_path=str(mock_yfinance / "taiex.csv"))
        assert (result["TAIEX_high"] >= result["TAIEX_low"]).all()

    def test_close_between_high_and_low(self, mock_yfinance):
        from src.scrape.scrape_taiex import scrape_taiex

        result = scrape_taiex(save_path=str(mock_yfinance / "taiex.csv"))
        assert (result["TAIEX_close"] >= result["TAIEX_low"]).all()
        assert (result["TAIEX_close"] <= result["TAIEX_high"]).all()


class TestScrapeTaiexEmptyResponse:
    def test_returns_empty_df_when_yfinance_empty(self, tmp_path):
        with patch("src.scrape.scrape_taiex.yf.Ticker") as MockTicker:
            MockTicker.return_value.history.return_value = pd.DataFrame()
            from src.scrape.scrape_taiex import scrape_taiex

            result = scrape_taiex(save_path=str(tmp_path / "taiex.csv"))
            assert isinstance(result, pd.DataFrame)
            assert result.empty


class TestScrapeTaiexPersistence:
    def test_csv_is_created(self, mock_yfinance):
        from src.scrape.scrape_taiex import scrape_taiex

        path = str(mock_yfinance / "taiex.csv")
        scrape_taiex(save_path=path)
        assert os.path.exists(path)

    def test_csv_matches_returned_dataframe(self, mock_yfinance):
        from src.scrape.scrape_taiex import scrape_taiex

        path = str(mock_yfinance / "taiex.csv")
        result = scrape_taiex(save_path=path)
        on_disk = pd.read_csv(path)
        pd.testing.assert_frame_equal(
            result.reset_index(drop=True),
            on_disk.reset_index(drop=True),
            check_dtype=False,
        )
