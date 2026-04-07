import pandas as pd
from fredapi import Fred
from dotenv import load_dotenv
import os


def fetch_macro_data(api_key: str, save_path: str = "../data/macro_fred.csv"):
    fred = Fred(api_key=api_key)

    cpi = fred.get_series("CPIAUCSL").reset_index()
    cpi.columns = ["date", "CPI"]

    gdp = fred.get_series("GDP").reset_index()
    gdp.columns = ["date", "GDP"]

    # Filter to 2005
    cpi = cpi[(cpi["date"] >= "2005-01-01") & (cpi["date"] <= "2005-12-31")]
    gdp = gdp[(gdp["date"] >= "2005-01-01") & (gdp["date"] <= "2005-12-31")]

    # Resample GDP to monthly before extracting year/month to ensure we have a value for each month
    gdp = gdp.set_index("date").resample("ME").ffill().reset_index()

    # Extract year and month for merging
    cpi["Year"] = cpi["date"].dt.year
    cpi["Month"] = cpi["date"].dt.month
    gdp["Year"] = gdp["date"].dt.year
    gdp["Month"] = gdp["date"].dt.month

    # Drop date before merging
    cpi = cpi.drop(columns=["date"])
    gdp = gdp.drop(columns=["date"])

    macro_df = pd.merge(cpi, gdp, on=["Year", "Month"], how="inner")

    macro_df.to_csv(save_path, index=False)
    print(f"FRED macro saved: {macro_df.shape}")
    return macro_df


if __name__ == "__main__":
    load_dotenv()

    fetch_macro_data(os.getenv("FRED_API_KEY"))
