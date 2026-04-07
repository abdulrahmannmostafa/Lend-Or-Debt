import logging
import os
import pandas as pd

# Logging
# ----------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)

# Single source of truth mapping each calendar month to its corresponding columns in the Taiwan dataset.
# PAY_1 does NOT exist in the UCI dataset — known upstream data issue
# The dataset jumps directly from PAY_0 (September) to PAY_2 (August).
# every borrower's rows run chronologically: iloc[0]=Apr, iloc[-1]=Sep.
# Tuple format: (calendar_month, pay_col, bill_col, payamt_col)
MONTH_PANEL = [
    (4, "PAY_6", "BILL_AMT6", "PAY_AMT6"),  # April 2005
    (5, "PAY_5", "BILL_AMT5", "PAY_AMT5"),  # May 2005
    (6, "PAY_4", "BILL_AMT4", "PAY_AMT4"),  # June 2005
    (7, "PAY_3", "BILL_AMT3", "PAY_AMT3"),  # July 2005
    (8, "PAY_2", "BILL_AMT2", "PAY_AMT2"),  # August 2005
    (9, "PAY_0", "BILL_AMT1", "PAY_AMT1"),  # September 2005
]


# Loaders
# ---------------------------------------------------------------------------


def _load_taiwan(path: str) -> pd.DataFrame:
    """
    Load the UCI Taiwan credit card Excel file to a DataFrame
    """
    log.info("Loading Taiwan dataset: %s", path)

    df = pd.read_excel(
        path, header=1
    )  # To delete the first row which is a duplicate header row in the original file
    df = df.reset_index(
        drop=True
    )  # Reset index to ensure positional alignment for panel building

    if (
        "ID" not in df.columns
    ):  # We need ID for merging the dataset based on the panel mapping with the scrapped CSVs
        log.warning("No 'ID' column found — generating sequential IDs")
        df.insert(0, "ID", range(1, len(df) + 1))

    log.info("Taiwan loaded: %s rows × %s cols", *df.shape)
    return df


def _load_macro_sources(
    macro_path: str,
    taiex_path: str,
    cbc_path: str,
    unemploy_path: str,
) -> tuple:
    """
    Load the four macro CSV files and return them as a tuple
    """
    log.info("Loading macro sources...")

    macro = pd.read_csv(macro_path)  # FRED
    taiex = pd.read_csv(taiex_path)  # Yahoo
    cbc = pd.read_csv(cbc_path)  # CBC Annual Report
    unemploy = pd.read_csv(unemploy_path)  # DGBAS

    log.info("FRED macro:    %s rows × %s cols", *macro.shape)
    log.info("TAIEX:         %s rows × %s cols", *taiex.shape)
    log.info("CBC rates:     %s rows × %s cols", *cbc.shape)
    log.info("Unemployment:  %s rows × %s cols", *unemploy.shape)

    return macro, taiex, cbc, unemploy


# Panel builder
# ---------------------------------------------------------------------------


def _build_panel(taiwan: pd.DataFrame) -> pd.DataFrame:
    """
    Melt the wide-format Taiwan dataset into a long-format panel.

    Each of the 30,000 borrowers produces 6 rows — one per observation month
    (April through September 2005).  Expected output: 180,000 rows x 6 cols.

    WHY THIS STEP IS NECESSARY:
    The macro time-series files (CPI, TAIEX, etc.) have one value per month.
    If we merge on the borrower level directly, every borrower gets the same
    single value regardless of timing — zero variance, zero predictive power.
    By melting to long format first, each borrower-month row can receive the
    exact macro value for that specific calendar month.

    The panel is sorted by [ID, Month] immediately so that within every
    borrower group, rows always run April -> September in order for ease of merging
    """
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

    panel = pd.concat(records, ignore_index=True)
    panel = panel.sort_values(["ID", "Month"]).reset_index(drop=True)

    log.info("Panel built: %s rows x %s cols (expected 180,000 x 6)", *panel.shape)
    return panel


# Macro merge
# ---------------------------------------------------------------------------


def _merge_macro(
    panel: pd.DataFrame,
    macro: pd.DataFrame,
    taiex: pd.DataFrame,
    cbc: pd.DataFrame,
    unemploy: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge all four macro sources onto the long-format panel using [Year, Month] as the join key
    """
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


# PAY_STATUS recoding
# ---------------------------------------------------------------------------


def _recode_pay_status(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Add PAY_STATUS_clean by clipping PAY_STATUS to a floor of 0

    UCI repayment status encoding:
        -2  no consumption  (account unused this month)
        -1  paid in full    (entire balance cleared)
         0  revolving       (minimum payment made, balance carried forward)
         1  delay 1 month
         ...
         8  delay 8 months

    WHY CLIPPING IS NECESSARY FOR AGGREGATIONS:
    sum() and max() treat -2 and -1 as numerically negative, which corrupts
    delinquency aggregations:
        - A borrower with five -2 months and one +3 delay gets a lower sum
          than one with six 0 months, making the risky borrower appear safer
        - max() on [-2, -1, -1, -1, -1, -1] returns -1, suggesting the
        borrower's "worst" month was a paid-in-full month

    Clipping maps -2 and -1 -> 0 (no delinquency) and leaves 1–8 unchanged
    Now sum() = total months delayed and max() = worst single-month delay

    """
    panel = panel.copy()
    panel["PAY_STATUS_clean"] = panel["PAY_STATUS"].clip(
        lower=0
    )  # This column to merge easily
    return panel


# Behaviour aggregation
# ---------------------------------------------------------------------------


def _aggregate_behaviour(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate the long-format panel back to one row per borrower

    This is not a data transformation step — it is an aggregation step only:
    Only behaviour aggregations that are direct summaries of the raw panel
    data belong here.  These are not analytical decisions, they are just
    collapsing 6 rows per borrower into a single representative value

    (N.B) They all have the same means across borrowers:
    Because every borrower shares the same 6 months, these averages will be
    identical across all 30,000 borrowers — zero variance.  They are included
    here so the validation step can verify their plausibility against known
    2005 Taiwan ranges.  The transformation step will drop pure averages and
    replace them with interaction features that have good per-borrower variance.
    """
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


# Main pipeline
# ---------------------------------------------------------------------------


def build_final_dataset(
    taiwan_path: str = "../data/default of credit card clients.xls",
    macro_path: str = "../data/macro_fred.csv",
    taiex_path: str = "../data/taiex_2005.csv",
    cbc_path: str = "../data/cbc_rates_2005.csv",
    unemploy_path: str = "../data/unemployment_2005.csv",
    save_path: str = "../data/taiwan_merged.csv",
) -> pd.DataFrame:
    """
    Data acquisition pipeline — builds and saves the raw merged dataset

    Pipeline steps
    --------------
    1.  Load Taiwan wide-format data          (30,000 x 25 expected)
    2.  Extract static borrower frame         (all original PAY_*, BILL_AMT*, PAY_AMT*, demographics, target)
    3.  Melt into long-format panel           (180,000 x 6 expected)
    4.  Merge four macro sources on           [Year, Month]
    5.  Sort panel by [ID, Month]             (logically per borrower)
    6.  Add PAY_STATUS_clean                  (clipped to 0 floor)
    7.  Aggregate panel -> behaviour features (one row per borrower)
    8.  Merge engineered features onto static frame (N.B: adds columns, not replaces originals)
    9.  Drop ID (join key only) and save
    """

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
