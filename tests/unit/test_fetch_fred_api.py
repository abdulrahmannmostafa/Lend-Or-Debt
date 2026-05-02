import os
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch
import src.scrape.fetch_fred_api as fred_module


def _make_full_cpi_series():
    dates = pd.date_range("2004-01-01", "2006-12-01", freq="MS")
    return pd.Series([100.0 + i * 0.2 for i in range(len(dates))], index=dates)


def _make_full_gdp_series():
    dates = pd.date_range("2004-01-01", "2006-10-01", freq="QS")
    return pd.Series([12000.0 + i * 50 for i in range(len(dates))], index=dates)


@pytest.fixture()
def mock_fred(tmp_path):
    with patch("fredapi.Fred") as MockFred:
        instance = MockFred.return_value
        instance.get_series.side_effect = lambda series_id: (
            _make_full_cpi_series()
            if series_id == "CPIAUCSL"
            else _make_full_gdp_series()
        )
        yield instance, tmp_path


class TestFetchMacroDataSchema:
    def test_returns_dataframe(self, mock_fred):
        instance, tmp = mock_fred
        from src.scrape.fetch_fred_api import fetch_macro_data

        result = fetch_macro_data(api_key="fake", save_path=str(tmp / "macro.csv"))
        assert isinstance(result, pd.DataFrame)

    def test_has_required_columns(self, mock_fred):
        instance, tmp = mock_fred
        from src.scrape.fetch_fred_api import fetch_macro_data

        result = fetch_macro_data(api_key="fake", save_path=str(tmp / "macro.csv"))
        assert {"Year", "Month", "CPI", "GDP"}.issubset(result.columns)

    def test_no_duplicate_month_year_rows(self, mock_fred):
        instance, tmp = mock_fred
        from src.scrape.fetch_fred_api import fetch_macro_data

        result = fetch_macro_data(api_key="fake", save_path=str(tmp / "macro.csv"))
        assert not result.duplicated(subset=["Year", "Month"]).any()


class TestFetchMacroDataFiltering:
    def test_only_2005_rows(self, mock_fred):
        instance, tmp = mock_fred
        from src.scrape.fetch_fred_api import fetch_macro_data

        result = fetch_macro_data(api_key="fake", save_path=str(tmp / "macro.csv"))
        assert (result["Year"] == 2005).all()

    def test_months_in_valid_range(self, mock_fred):
        instance, tmp = mock_fred
        from src.scrape.fetch_fred_api import fetch_macro_data

        result = fetch_macro_data(api_key="fake", save_path=str(tmp / "macro.csv"))
        assert result["Month"].between(1, 12).all()

    def test_no_null_values(self, mock_fred):
        instance, tmp = mock_fred
        from src.scrape.fetch_fred_api import fetch_macro_data

        result = fetch_macro_data(api_key="fake", save_path=str(tmp / "macro.csv"))
        assert not result.isnull().any().any()


class TestFetchMacroDataPersistence:
    def test_csv_is_created(self, mock_fred):
        instance, tmp = mock_fred
        from src.scrape.fetch_fred_api import fetch_macro_data

        path = str(tmp / "macro.csv")
        fetch_macro_data(api_key="fake", save_path=path)
        assert os.path.exists(path)

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
