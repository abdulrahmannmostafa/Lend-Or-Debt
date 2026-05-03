import pandas as pd
from dotenv import load_dotenv
import os


# I made this to fetch the macroeconomic data from FRED API, processes it and saves it as a CSV file
def fetch_macro_data(api_key: str, save_path: str = "../data/macro_fred.csv"):
    from fredapi import Fred

    # here i instantiate the FRED API client using the provided API key
    fred = Fred(api_key=api_key)

    # i get the Consumer Price Index for All Urban Consumers (monthly data)
    # reset index to convert the Series to a DataFrame and rename columns for clarity
    cpi = fred.get_series("CPIAUCSL").reset_index()

    # i rename the columns to "date" and "CPI" for better readability
    cpi.columns = ["date", "CPI"]

    # i get the Gross Domestic Product (quarterly data)
    gdp = fred.get_series("GDP").reset_index()
    gdp.columns = ["date", "GDP"]

    # Filter to 2005
    # Both DataFrames are sliced to only include rows where the date falls within 2005
    # String comparison works here because pandas automatically parses ISO date strings
    cpi = cpi[(cpi["date"] >= "2005-01-01") & (cpi["date"] <= "2005-12-31")]
    gdp = gdp[(gdp["date"] >= "2005-01-01") & (gdp["date"] <= "2005-12-31")]

    # GDP is quarterly and CPI is monthly, so we need to forward fill the GDP values to match the monthly frequency of CPI
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
    api_key = os.getenv("FRED_API_KEY")

    if not api_key:
        raise ValueError("FRED API key is required in the .env file")
    fetch_macro_data(api_key=api_key)
