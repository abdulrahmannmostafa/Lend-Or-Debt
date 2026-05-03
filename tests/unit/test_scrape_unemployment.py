import os
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

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

    # I mock the requests.get function to return a fake response containing our MOCK_XML data
    # it simulates the behavior of the real HTTP request that the scrape_unemployment function would make to fetch the unemployment data
    with patch("src.scrape.scrape_unemployment.requests.get") as mock_get:
        resp = MagicMock()
        resp.status_code = 200
        resp.content = MOCK_XML
        resp.text = MOCK_XML.decode("utf-8")
        mock_get.return_value = resp

        # yield the temporary path for use in tests, allowing the test to run with the mock in place and then automatically undoing the patch after the test completes
        yield tmp_path


class TestScrapeUnemploymentSchema:
    # I check that the function returns a DataFrame to ensure that the output is in the expected format for downstream analysis
    def test_returns_dataframe(self, mock_http):
        from src.scrape.scrape_unemployment import scrape_unemployment

        result = scrape_unemployment(save_path=str(mock_http / "unemp.csv"))

        assert isinstance(result, pd.DataFrame)

    # I check that the required columns are present in the DataFrame to ensure that the data is structured correctly for downstream analysis
    def test_required_columns_present(self, mock_http):
        from src.scrape.scrape_unemployment import scrape_unemployment

        result = scrape_unemployment(save_path=str(mock_http / "unemp.csv"))

        for col in ["Year", "Month", "Unemployment_Rate", "Source_Period"]:
            assert col in result.columns

    # I check that there are no null values in the DataFrame to ensure data integrity and that the parsing process successfully extracted all necessary information from the XML
    def test_no_nulls(self, mock_http):
        from src.scrape.scrape_unemployment import scrape_unemployment

        result = scrape_unemployment(save_path=str(mock_http / "unemp.csv"))

        assert not result.isnull().any().any()


class TestScrapeUnemploymentMapping:
    # I check that the DataFrame contains exactly 6 rows, which corresponds to the 6 months of data (April–September) that we expect to extract from the XML
    def test_exactly_6_rows(self, mock_http):
        from src.scrape.scrape_unemployment import scrape_unemployment

        result = scrape_unemployment(save_path=str(mock_http / "unemp.csv"))

        assert len(result) == 6

    # I check that the Month column contains the correct month numbers (4-9) to ensure that the data is correctly filtered to include only the relevant months from the XML
    def test_correct_months(self, mock_http):
        from src.scrape.scrape_unemployment import scrape_unemployment

        result = scrape_unemployment(save_path=str(mock_http / "unemp.csv"))

        assert set(result["Month"]) == {4, 5, 6, 7, 8, 9}

    # I check that for April–June, the unemployment rate is 4.13% to verify that the function correctly parsed the pre-hike rate from the XML
    def test_h1_rate_applied_to_april_june(self, mock_http):
        from src.scrape.scrape_unemployment import scrape_unemployment

        result = scrape_unemployment(save_path=str(mock_http / "unemp.csv"))

        h1 = result[result["Month"].isin([4, 5, 6])]["Unemployment_Rate"]

        assert np.allclose(h1.to_numpy(), 4.13)

    # I check that for July–September, the unemployment rate is 4.05% to verify that the function correctly parsed the post-hike rate from the XML
    def test_h2_rate_applied_to_july_september(self, mock_http):
        from src.scrape.scrape_unemployment import scrape_unemployment

        result = scrape_unemployment(save_path=str(mock_http / "unemp.csv"))

        h2 = result[result["Month"].isin([7, 8, 9])]["Unemployment_Rate"]

        assert np.allclose(h2.to_numpy(), 4.05)

    # I check that the unemployment rates are positive to ensure that the parsed values are reasonable and that there were no errors in the parsing process that resulted in invalid negative rates
    def test_rates_are_positive(self, mock_http):
        from src.scrape.scrape_unemployment import scrape_unemployment

        result = scrape_unemployment(save_path=str(mock_http / "unemp.csv"))

        assert (result["Unemployment_Rate"] > 0).all()

    # I check that the Year column is 2005 for all rows to confirm that the data corresponds to the correct year as specified in the XML
    def test_year_is_2005(self, mock_http):
        from src.scrape.scrape_unemployment import scrape_unemployment

        result = scrape_unemployment(save_path=str(mock_http / "unemp.csv"))

        assert (result["Year"] == 2005).all()


class TestScrapeUnemploymentPersistence:
    # I check that the function creates a CSV file at the specified path to verify that the data is being saved correctly for future use
    def test_csv_is_created(self, mock_http):
        from src.scrape.scrape_unemployment import scrape_unemployment

        path = str(mock_http / "unemp.csv")

        scrape_unemployment(save_path=path)

        assert os.path.exists(path)

    # I check that the content of the CSV file matches the DataFrame returned by the function to ensure that the data is being saved correctly and consistently
    def test_csv_matches_return(self, mock_http):
        from src.scrape.scrape_unemployment import scrape_unemployment

        path = str(mock_http / "unemp.csv")

        result = scrape_unemployment(save_path=path)

        on_disk = pd.read_csv(path)

        # reset_index is used to ignore the index when comparing the DataFrames, and check_dtype=False allows for comparison even if the data types differ
        # because I have no control over the data types that will be inferred when reading from CSV, but I want to ensure that the values themselves match regardless of type differences
        pd.testing.assert_frame_equal(
            result.reset_index(drop=True),
            on_disk.reset_index(drop=True),
            check_dtype=False,
        )
