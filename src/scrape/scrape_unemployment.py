import requests
import pandas as pd

# This library parses the XML data into navigable tree structure
import xml.etree.ElementTree as ET

import urllib3

# When I make HTTPS requests with verify=False (disabled SSL certificate verification), urllib3 normally prints a warning for every request, so this line silences those warnings
# It's used here because the DGBAS government server has SSL issues that would otherwise cause the request to fail entirely
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def scrape_unemployment(save_path: str = "../data/unemployment_2005.csv"):

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

    # Loop through each record in the XML (That tag contains the unemployment rate data by county)
    for record in root.findall("縣市別失業率"):
        # Extract the period (like "2005Jan.-June", "2005July-Dec.")
        period = record.findtext("年月別_Year_and_month", default="").strip()

        # Extract the unemployment rate for Taiwan area
        # I made default=None to ignore records that don't have the Taiwan rate
        taiwan_rate = record.findtext("臺灣地區_Taiwan_Area_百分比", default=None)

        records.append({"period": period, "Unemployment_Rate": taiwan_rate})

    df_all = pd.DataFrame(records)
    print("\nTotal records in XML: {len(df_all)}")

    # Filter to 2005 semi annual periods only to ensure we have the correct mapping for April-September 2005
    df_2005 = df_all[df_all["period"].str.startswith("2005")].copy()

    print("\n2005 records found:")
    print(df_2005.to_string(index=False))

    # The DGBAS data is semi-annual (every 6 months), not monthly
    # So there are only two rates for 2005 -> rate_h1 and rate_h2

    # Map semi-annual periods to individual months
    # 2005Jan.-June  -> covers April (4), May (5), June (6)
    # 2005July-Dec.  -> covers July (7), August (8), September (9)

    # .values[0] extracts the scalar value from the filtered Series — without it we'd get a one element array instead of a plain number like [2.5] instead of 2.5
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
