import os
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

# Corrected XML: use real Unicode string, then encode to bytes
MOCK_XML = """<?xml version="1.0" encoding="UTF-8"?>
<root>
  <縣市別失業率>
    <年月別_Year_and_month>2005Jan.-June</年月別_Year_and_month>
    <臺灣地區_Taiwan_Area_百分比>4.13</臺灣地區_Taiwan_Area_百分比>
  </縣市別失業率>
  <縣市別失業率>
    <年月別_Year_and_month>2005July-Dec.</年月別_Year_and_month>
    <臺灣地區_Taiwan_Area_百分比>4.05</臺灣地區_Taiwan_Area_百分比>
  </縣市別失業率>
  <縣市別失業率>
    <年月別_Year_and_month>2004Jan.-June</年月別_Year_and_month>
    <臺灣地區_Taiwan_Area_百分比>4.50</臺灣地區_Taiwan_Area_百分比>
  </縣市別失業率>
</root>
""".encode("utf-8")


@pytest.fixture()
def mock_http(tmp_path):
    with patch("src.scrape.scrape_unemployment.requests.get") as mock_get:
        resp = MagicMock()
        resp.status_code = 200
        resp.content = MOCK_XML
        resp.text = MOCK_XML.decode("utf-8")
        mock_get.return_value = resp
        yield tmp_path


class TestScrapeUnemploymentSchema:
    def test_returns_dataframe(self, mock_http):
        from src.scrape.scrape_unemployment import scrape_unemployment

        result = scrape_unemployment(save_path=str(mock_http / "unemp.csv"))
        assert isinstance(result, pd.DataFrame)

    def test_required_columns_present(self, mock_http):
        from src.scrape.scrape_unemployment import scrape_unemployment

        result = scrape_unemployment(save_path=str(mock_http / "unemp.csv"))
        for col in ["Year", "Month", "Unemployment_Rate", "Source_Period"]:
            assert col in result.columns

    def test_no_nulls(self, mock_http):
        from src.scrape.scrape_unemployment import scrape_unemployment

        result = scrape_unemployment(save_path=str(mock_http / "unemp.csv"))
        assert not result.isnull().any().any()


class TestScrapeUnemploymentMapping:
    def test_exactly_6_rows(self, mock_http):
        from src.scrape.scrape_unemployment import scrape_unemployment

        result = scrape_unemployment(save_path=str(mock_http / "unemp.csv"))
        assert len(result) == 6

    def test_correct_months(self, mock_http):
        from src.scrape.scrape_unemployment import scrape_unemployment

        result = scrape_unemployment(save_path=str(mock_http / "unemp.csv"))
        assert set(result["Month"]) == {4, 5, 6, 7, 8, 9}

    def test_h1_rate_applied_to_april_june(self, mock_http):
        from src.scrape.scrape_unemployment import scrape_unemployment

        result = scrape_unemployment(save_path=str(mock_http / "unemp.csv"))
        h1 = result[result["Month"].isin([4, 5, 6])]["Unemployment_Rate"]
        assert np.allclose(h1.to_numpy(), 4.13)

    def test_h2_rate_applied_to_july_september(self, mock_http):
        from src.scrape.scrape_unemployment import scrape_unemployment

        result = scrape_unemployment(save_path=str(mock_http / "unemp.csv"))
        h2 = result[result["Month"].isin([7, 8, 9])]["Unemployment_Rate"]
        assert np.allclose(h2.to_numpy(), 4.05)

    def test_rates_are_positive(self, mock_http):
        from src.scrape.scrape_unemployment import scrape_unemployment

        result = scrape_unemployment(save_path=str(mock_http / "unemp.csv"))
        assert (result["Unemployment_Rate"] > 0).all()

    def test_year_is_2005(self, mock_http):
        from src.scrape.scrape_unemployment import scrape_unemployment

        result = scrape_unemployment(save_path=str(mock_http / "unemp.csv"))
        assert (result["Year"] == 2005).all()


class TestScrapeUnemploymentPersistence:
    def test_csv_is_created(self, mock_http):
        from src.scrape.scrape_unemployment import scrape_unemployment

        path = str(mock_http / "unemp.csv")
        scrape_unemployment(save_path=path)
        assert os.path.exists(path)

    def test_csv_matches_return(self, mock_http):
        from src.scrape.scrape_unemployment import scrape_unemployment

        path = str(mock_http / "unemp.csv")
        result = scrape_unemployment(save_path=path)
        on_disk = pd.read_csv(path)

        pd.testing.assert_frame_equal(
            result.reset_index(drop=True),
            on_disk.reset_index(drop=True),
            check_dtype=False,
        )
