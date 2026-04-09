import requests
import pandas as pd
import xml.etree.ElementTree as ET
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def scrape_unemployment(save_path: str = "../data/unemployment_2005.csv"):
    """
    Taiwan unemployment rate for April-September 2005.
    Source: Directorate-General of Budget, Accounting and Statistics (DGBAS)
    Dataset: Unemployment Rate by County — Human Resources Survey
    Portal: https://data.gov.tw/en/datasets/6640
    Direct XML: https://ws.dgbas.gov.tw/001/Upload/461/relfile/11525/230038/mp0101a10.xml

    Note: Data is published as semi-annual averages (Jan-June, July-Dec).
    April-June 2005 mapped from 2005Jan.-June value.
    July-September 2005 mapped from 2005July-Dec. value.
    SSL verify=False used due to Taiwanese government server certificate chain issue.
    """

    xml_url = (
        "https://ws.dgbas.gov.tw/001/Upload/461/relfile/11525/230038/mp0101a10.xml"
    )
    headers = {"User-Agent": "Mozilla/5.0"}

    print("Downloading unemployment XML from DGBAS open data portal...")
    response = requests.get(xml_url, headers=headers, timeout=30, verify=False)
    print("Status: {response.status_code} | Size: {len(response.content)} bytes")

    # Parse XML
    root = ET.fromstring(response.content)

    # Extract all records into a flat list
    records = []
    for record in root.findall(
        "縣市別失業率"
    ):  # Loop through each record in the XML (That tag contains the unemployment rate data by county)
        period = record.findtext(
            "年月別_Year_and_month", default=""
        ).strip()  # Extract the period (e.g., "2005Jan.-June", "2005July-Dec.")
        taiwan_rate = record.findtext(
            "臺灣地區_Taiwan_Area_百分比", default=None
        )  # Extract the unemployment rate for Taiwan area
        records.append({"period": period, "Unemployment_Rate": taiwan_rate})

    df_all = pd.DataFrame(records)
    print("\nTotal records in XML: {len(df_all)}")

    # Filter to 2005 semi annual periods only to ensure we have the correct mapping for April-September 2005
    df_2005 = df_all[df_all["period"].str.startswith("2005")].copy()
    print("\n2005 records found:")
    print(df_2005.to_string(index=False))

    # Map semi-annual periods to individual months
    # 2005Jan.-June  → covers April (4), May (5), June (6)
    # 2005July-Dec.  → covers July (7), August (8), September (9)
    rate_h1 = df_2005[df_2005["period"] == "2005Jan.-June"]["Unemployment_Rate"].values[
        0
    ]
    rate_h2 = df_2005[df_2005["period"] == "2005July-Dec."]["Unemployment_Rate"].values[
        0
    ]

    print("\n2005 H1 (Jan-June) rate: {rate_h1}%")
    print("2005 H2 (July-Dec) rate: {rate_h2}%")

    # Build monthly dataframe
    monthly_data = {
        "Year": [2005, 2005, 2005, 2005, 2005, 2005],
        "Month": [4, 5, 6, 7, 8, 9],
        "Unemployment_Rate": [
            float(rate_h1),  # April
            float(rate_h1),  # May
            float(rate_h1),  # June
            float(rate_h2),  # July
            float(rate_h2),  # August
            float(rate_h2),  # September
        ],
        "Source_Period": [
            "2005Jan.-June",
            "2005Jan.-June",
            "2005Jan.-June",
            "2005July-Dec.",
            "2005July-Dec.",
            "2005July-Dec.",
        ],
    }

    df_monthly = pd.DataFrame(monthly_data)

    df_monthly.to_csv(save_path, index=False)

    print("\nUnemployment data saved: {df_monthly.shape} -> {save_path}")
    print(df_monthly.to_string(index=False))

    return df_monthly


if __name__ == "__main__":
    scrape_unemployment()
