import yfinance as yf
import pandas as pd


def scrape_taiex(save_path: str = "../data/taiex_2005.csv"):
    """
    Downloads TAIEX (^TWII) monthly historical data for April-September 2005
    from Yahoo Finance using the yfinance library.
    Source: Yahoo Finance — Taiwan Capitalization Weighted Stock Index (^TWII)
    URL: https://finance.yahoo.com/quote/%5ETWII/history/
    """

    print("Downloading TAIEX data from Yahoo Finance...")

    # Download daily data for the full 2005 period
    ticker = yf.Ticker("^TWII")
    df_daily = ticker.history(start="2005-04-01", end="2005-09-30")

    if df_daily.empty:
        print("ERROR: No data returned from Yahoo Finance.")
        return pd.DataFrame()

    print(f"Daily data downloaded: {df_daily.shape}")
    print(df_daily.head())

    # Reset index to access the date
    df_daily = df_daily.reset_index()
    df_daily["Year"] = df_daily["Date"].dt.year
    df_daily["Month"] = df_daily["Date"].dt.month

    # Aggregate to monthly — use last closing value of each month because it reflects the market conditions at the end of the month, which is more relevant for credit card data observed at the end of month
    taiex_monthly = (
        df_daily.groupby(["Year", "Month"])
        .agg(
            TAIEX_open=("Open", "first"),
            TAIEX_high=("High", "max"),
            TAIEX_low=("Low", "min"),
            TAIEX_close=("Close", "last"),
            TAIEX_avg=("Close", "mean"),
        )
        .reset_index()
    )

    print(f"\nMonthly TAIEX data:")
    print(taiex_monthly)

    taiex_monthly.to_csv(save_path, index=False)
    print(f"\nTAIEX saved: {taiex_monthly.shape} -> {save_path}")

    return taiex_monthly


if __name__ == "__main__":
    scrape_taiex()
