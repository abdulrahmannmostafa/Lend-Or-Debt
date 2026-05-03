import logging
import pandas as pd

# I made that script to combine Taiwan Credit Card dataset with the macroeconomic data from FRED, TAIEX, CBC and DGBAS to build a final CSV
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)

# Mapper for the month panel construction: each tuple corresponds to one month and specifies the relevant columns for that month
MONTH_PANEL = [
    (4, "PAY_6", "BILL_AMT6", "PAY_AMT6"),  # April 2005
    (5, "PAY_5", "BILL_AMT5", "PAY_AMT5"),  # May 2005
    (6, "PAY_4", "BILL_AMT4", "PAY_AMT4"),  # June 2005
    (7, "PAY_3", "BILL_AMT3", "PAY_AMT3"),  # July 2005
    (8, "PAY_2", "BILL_AMT2", "PAY_AMT2"),  # August 2005
    (9, "PAY_0", "BILL_AMT1", "PAY_AMT1"),  # September 2005
]


def _load_taiwan(path: str) -> pd.DataFrame:
    log.info("Loading Taiwan dataset: %s", path)

    # To delete the first row which is a duplicate header row in the original file
    df = pd.read_excel(path, header=1)

    # Reset index to ensure positional alignment for panel building
    df = df.reset_index(drop=True)

    # We need ID for merging the dataset based on the panel mapping with the scrapped CSVs for that we need to generate sequential IDs if not present
    if "ID" not in df.columns:
        log.warning("No 'ID' column found — generating sequential IDs")

        # Insert ID column at the front with sequential integers starting from 1
        df.insert(0, "ID", range(1, len(df) + 1))

    log.info("Taiwan loaded: %s rows × %s cols", *df.shape)
    return df


def _load_macro_sources(
    macro_path: str,
    taiex_path: str,
    cbc_path: str,
    unemploy_path: str,
) -> tuple:
    log.info("Loading macro sources...")

    macro = pd.read_csv(macro_path)  # FRED
    taiex = pd.read_csv(taiex_path)  # Yahoo
    cbc = pd.read_csv(cbc_path)  # CBC Annual Report
    unemploy = pd.read_csv(unemploy_path)  # DGBAS

    log.info("FRED macro:    %s rows x %s cols", *macro.shape)
    log.info("TAIEX:         %s rows x %s cols", *taiex.shape)
    log.info("CBC rates:     %s rows x %s cols", *cbc.shape)
    log.info("Unemployment:  %s rows x %s cols", *unemploy.shape)

    return macro, taiex, cbc, unemploy


# I built that to convert the wide format of Taiwan dataset from having one row per borrower with multiple columns for each month
# to a long format panel with one row per borrower per month
def _build_panel(taiwan: pd.DataFrame) -> pd.DataFrame:
    log.info("Building long-format panel...")
    records = []

    for month_num, pay_col, bill_col, payamt_col in MONTH_PANEL:
        tmp = pd.DataFrame(
            {
                "ID": taiwan["ID"],
                "Year": 2005,
                "Month": month_num,
                "PAY_STATUS": taiwan[pay_col],  # repayment status this month
                "BILL_AMT": taiwan[bill_col],  # bill statement this month
                "PAY_AMT": taiwan[payamt_col],  # actual payment this month
            }
        )
        records.append(tmp)

    # For each of the 6 months in MONTH_PANEL
    # it creates a temporary DataFrame with the relevant columns, then stacks all 6 DataFrames vertically
    # With 30,000 borrowers × 6 months = 180,000 rows

    # ignore index to get a clean sequential index after concatenation like 0, 1, 2, ..., 179999
    panel = pd.concat(records, ignore_index=True)

    # Sort by ID and Month to ensure the panel is ordered correctly for later processing
    panel = panel.sort_values(["ID", "Month"]).reset_index(drop=True)

    log.info("Panel built: %s rows x %s cols (expected 180,000 x 6)", *panel.shape)
    return panel


# I built that to merge the macroeconomic data from FRED, TAIEX, CBC and DGBAS onto the long format panel based on the Year and Month columns
# A left merge ensures all panel rows are kept even if a macro source has no matching month
# Each merge adds new columns without dropping existing rows
def _merge_macro(
    panel: pd.DataFrame,
    macro: pd.DataFrame,
    taiex: pd.DataFrame,
    cbc: pd.DataFrame,
    unemploy: pd.DataFrame,
) -> pd.DataFrame:
    log.info("Merging macro sources onto panel...")

    panel = panel.merge(
        macro[["Year", "Month", "CPI", "GDP"]],
        on=["Year", "Month"],
        how="left",
    )
    log.info("After FRED merge:          %s rows x %s cols", *panel.shape)

    panel = panel.merge(
        taiex[["Year", "Month", "TAIEX_close"]],
        on=["Year", "Month"],
        how="left",
    )
    log.info("After TAIEX merge:         %s rows x %s cols", *panel.shape)

    panel = panel.merge(
        cbc[["Year", "Month", "Discount_Rate"]],
        on=["Year", "Month"],
        how="left",
    )
    log.info("After CBC merge:           %s rows x %s cols", *panel.shape)

    panel = panel.merge(
        unemploy[["Year", "Month", "Unemployment_Rate"]],
        on=["Year", "Month"],
        how="left",
    )
    log.info("After Unemployment merge:  %s rows x %s cols", *panel.shape)

    return panel


# The original PAY_STATUS column contains negative values (like -1, -2) that encode "paid on time" or "no consumption"
# sets any negative value to 0, making the column purely numeric for aggregation because negative delinquency doesn't make mathematical sense
def _recode_pay_status(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()
    panel["PAY_STATUS_clean"] = panel["PAY_STATUS"].clip(
        lower=0
    )  # This column to merge easily
    return panel


# I made this to collapse the long format panel back to one row per borrower by aggregating the monthly behaviour and macro features
def _aggregate_behaviour(panel: pd.DataFrame) -> pd.DataFrame:
    log.info("Aggregating panel to borrower level...")

    agg = (
        panel.groupby("ID", sort=True)
        .agg(
            avg_bill=("BILL_AMT", "mean"),
            avg_payment=("PAY_AMT", "mean"),
            max_delinquency=("PAY_STATUS_clean", "max"),
            total_delinquency=("PAY_STATUS_clean", "sum"),
            avg_macro_CPI=("CPI", "mean"),
            avg_macro_GDP=("GDP", "mean"),
            avg_macro_TAIEX=("TAIEX_close", "mean"),
            avg_macro_rate=("Discount_Rate", "mean"),
            avg_macro_unemp=("Unemployment_Rate", "mean"),
        )
        .reset_index()
    )

    log.info("Behaviour aggregation done: %s rows x %s cols", *agg.shape)
    return agg


def build_final_dataset(
    taiwan_path: str = "../data/default of credit card clients.xls",
    macro_path: str = "../data/macro_fred.csv",
    taiex_path: str = "../data/taiex_2005.csv",
    cbc_path: str = "../data/cbc_rates_2005.csv",
    unemploy_path: str = "../data/unemployment_2005.csv",
    save_path: str = "../data/taiwan_merged.csv",
) -> pd.DataFrame:

    # Step 1
    taiwan = _load_taiwan(taiwan_path)
    macro, taiex, cbc, unemploy = _load_macro_sources(
        macro_path, taiex_path, cbc_path, unemploy_path
    )

    # Step 2
    static_cols = [
        "ID",
        "LIMIT_BAL",
        "SEX",
        "EDUCATION",
        "MARRIAGE",
        "AGE",
        "PAY_0",
        "PAY_2",
        "PAY_3",
        "PAY_4",
        "PAY_5",
        "PAY_6",
        "BILL_AMT1",
        "BILL_AMT2",
        "BILL_AMT3",
        "BILL_AMT4",
        "BILL_AMT5",
        "BILL_AMT6",
        "PAY_AMT1",
        "PAY_AMT2",
        "PAY_AMT3",
        "PAY_AMT4",
        "PAY_AMT5",
        "PAY_AMT6",
        "default payment next month",
    ]
    static = taiwan[static_cols].copy()
    log.info("Static frame: %s rows x %s cols", *static.shape)

    # Step 3
    panel = _build_panel(taiwan)

    # Step 4
    panel = _merge_macro(panel, macro, taiex, cbc, unemploy)

    # Step 5
    panel = panel.sort_values(["ID", "Month"]).reset_index(drop=True)

    # Step 6
    panel = _recode_pay_status(panel)

    # Step 7
    behaviour = _aggregate_behaviour(panel)

    # Step 8
    final = static.merge(behaviour, on="ID", how="left")

    # Step 9
    final = final.drop(columns=["ID"])

    log.info("Final merged dataset: %s rows x %s cols", *final.shape)
    log.info("Columns: %s", final.columns.tolist())

    final.to_csv(save_path, index=False)
    log.info("Raw merged dataset saved: %s", save_path)

    return final


if __name__ == "__main__":
    build_final_dataset()
