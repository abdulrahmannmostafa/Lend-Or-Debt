import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import skew, kurtosis, kstest
from sklearn.ensemble import IsolationForest

# Logging
# ===========================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)


# Constants based on buisiness rules
# ===========================================================================

# 2005 Taiwan benchmark constants
# Sources:
#   CPI          — Taiwan DGBAS national statistics
#   GDP          — FRED RGDPNAT series (index level, not growth rate)
#   TAIEX        — TWSE historical archives
#   Discount     — CBC Annual Report 2005 (rate-hiking cycle)
#   Unemployment — DGBAS Human Resources Survey
#   Default rate — Yeh (2009), UCI dataset paper


CPI_MIN, CPI_MAX = 100, 200  # 2005 Taiwan CPI range from DGBAS data
GDP_MIN, GDP_MAX = 10_000.0, 15_000.0  # 2005 FRED RGDPNAT index range (not growth rate)
TAIEX_MIN, TAIEX_MAX = 5_000.0, 7_500.0  # 2005 TAIEX range from TWSE historical data
DISCOUNT_MIN, DISCOUNT_MAX = (
    1.5,
    2.5,
)  # 2005 CBC discount rate range from official records (two hikes in March and July)
UNEMPLOY_MIN, UNEMPLOY_MAX = (
    3.0,
    5.5,
)  # 2005 Taiwan unemployment rate range from DGBAS Human Resources Survey

EXPECTED_ROWS = 30_000
DEFAULT_RATE_MIN = 0.20  # 2005 Taiwan credit card default rate benchmark from Yeh (2009) UCI dataset paper
DEFAULT_RATE_MAX = 0.25  # 2005 Taiwan credit card default rate benchmark from Yeh (2009) UCI dataset paper

SEX_VALUES = {1, 2}
EDUCATION_VALUES = {1, 2, 3, 4}
MARRIAGE_VALUES = {1, 2, 3}
PAY_STATUS_RANGE = (-2, 8)
TARGET_VALUES = {0, 1}

EXPECTED_COLUMNS = [
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
    "avg_bill",
    "avg_payment",
    "max_delinquency",
    "total_delinquency",
    "avg_macro_CPI",
    "avg_macro_GDP",
    "avg_macro_TAIEX",
    "avg_macro_rate",
    "avg_macro_unemp",
]

# Continuous numeric columns used for outlier and distribution checks
NUMERIC_COLS = [
    "LIMIT_BAL",
    "AGE",
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
    "avg_bill",
    "avg_payment",
]


# I built that function to have a consistent way to format the results of each validation check across all dimensions
def _check(label: str, passed: bool, observed=None, expected=None) -> dict:
    return {
        "check": label,
        "success": bool(passed),
        "observed": str(observed) if observed is not None else "—",
        "expected": str(expected) if expected is not None else "—",
    }


# Dimension 1 — Accuracy
# "Does data correctly represent reality?"
# --------------------------------------------------------------


def validate_accuracy(df: pd.DataFrame) -> list:
    log.info("Validating Dimension 1: Accuracy")

    results = []

    # AGE must be a logical human age for a credit card holder from 18 to 100 years old
    if "AGE" in df.columns:
        bad = int(((df["AGE"] < 18) | (df["AGE"] > 100)).sum())
        results.append(
            _check(
                "AGE in [18, 100] (business rule: credit card eligibility floor)",
                bad == 0,
                observed=f"{bad} out-of-range",
                expected="0",
            )
        )

    # Credit limit must be strictly positive and below 1B NTD
    if "LIMIT_BAL" in df.columns:
        bad = int(((df["LIMIT_BAL"] <= 0) | (df["LIMIT_BAL"] > 1_000_000_000)).sum())
        results.append(
            _check(
                "LIMIT_BAL in (0, 1B NTD] (business rule: positive credit limit)",
                bad == 0,
                observed=f"{bad} out-of-range",
                expected="0",
            )
        )

    # Cross-check macro columns against published 2005 Taiwan benchmarks
    macro_checks = [
        (
            "avg_macro_CPI",
            CPI_MIN,
            CPI_MAX,
            "CPI in 2005 Taiwan published range [100, 200] (DGBAS data)",
        ),
        (
            "avg_macro_GDP",
            GDP_MIN,
            GDP_MAX,
            "GDP index in 2005 FRED range [10000, 15000]",
        ),
        (
            "avg_macro_TAIEX",
            TAIEX_MIN,
            TAIEX_MAX,
            "TAIEX in 2005 TWSE range [5000, 7500]",
        ),
        (
            "avg_macro_rate",
            DISCOUNT_MIN,
            DISCOUNT_MAX,
            "Discount Rate in 2005 CBC range [1.5, 2.5]",
        ),
        (
            "avg_macro_unemp",
            UNEMPLOY_MIN,
            UNEMPLOY_MAX,
            "Unemployment in 2005 DGBAS range [3.0, 5.5]",
        ),
    ]
    for col, lo, hi, label in macro_checks:
        if col not in df.columns:
            continue
        bad = int(((df[col] < lo) | (df[col] > hi)).sum())
        mean = round(float(df[col].mean()), 4)
        results.append(
            _check(
                label,
                bad == 0,
                observed=f"mean={mean}, {bad} out-of-range rows",
                expected=f"all values in [{lo}, {hi}]",
            )
        )

    return results


# Dimension 2 — Consistency
# "Is data uniform across systems, datasets, and time periods?"
# ===========================================================================


def validate_consistency(df: pd.DataFrame) -> list:
    log.info("Validating Dimension 2: Consistency")

    results = []

    pay_cols = [
        c
        for c in ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
        if c in df.columns
    ]

    # Zero total_delinquency -> no individual PAY_STATUS > 0
    if "total_delinquency" in df.columns and pay_cols:
        zero_mask = df["total_delinquency"] == 0
        any_positive = int((df.loc[zero_mask, pay_cols] > 0).any(axis=1).sum())
        results.append(
            _check(
                "Borrowers with total_delinquency=0 have no PAY_STATUS > 0 (logical cross-column)",
                any_positive == 0,
                observed=f"{any_positive} inconsistent rows",
                expected="0",
            )
        )

    # max_delinquency cannot exceed the PAY_STATUS ceiling of 8
    if "max_delinquency" in df.columns:
        over = int((df["max_delinquency"] > 8).sum())
        results.append(
            _check(
                "max_delinquency <= 8 (ceiling of PAY_STATUS scale)",
                over == 0,
                observed=f"{over} values above 8",
                expected="0",
            )
        )

    # Payments cannot be a negative cash flow
    if "avg_payment" in df.columns:
        neg = int((df["avg_payment"] < 0).sum())
        results.append(
            _check(
                "avg_payment >= 0 (payments are non-negative cash flows)",
                neg == 0,
                observed=f"{neg} negative values",
                expected="0",
            )
        )

    # Credit notes cannot exceed -1M NTD
    # I check if the value is below -1M (more negative than -1M) because credit notes are negative amounts
    # If the value is above -1M (less negative than -1M), that is consistent with reality because credit notes are typically smaller than 1M NTD in magnitude
    if "avg_bill" in df.columns:
        too_low = int((df["avg_bill"] < -1_000_000).sum())
        results.append(
            _check(
                "avg_bill > -1,000,000 NTD (credit note floor)",
                too_low == 0,
                observed=f"{too_low} below floor",
                expected="0",
            )
        )

    # PAY_AMT columns should never exceed BILL_AMT by an unreasonable factor
    # (overpayment of >10x average bill is a data quality signal)
    for i in range(1, 7):
        pa, ba = f"PAY_AMT{i}", f"BILL_AMT{i}"
        if pa not in df.columns or ba not in df.columns:
            continue
        # Only check rows where bill > 0 to avoid division-by-zero noise
        mask = df[ba] > 0
        if mask.sum() == 0:
            continue
        ratio = df.loc[mask, pa] / df.loc[mask, ba]
        excess = int((ratio > 10).sum())
        results.append(
            _check(
                f"{pa} / {ba} <= 10x (extreme overpayment check)",
                excess == 0,
                observed=f"{excess} rows with ratio > 10x",
                expected="0",
            )
        )

    return results


# Dimension 3 — Completeness
# "Is all the required data present?"


def validate_completeness(df: pd.DataFrame) -> list:
    log.info("Validating Dimension 3: Completeness")

    results = []

    # Row count vs expected
    results.append(
        _check(
            "Row count = 30,000 (data size vs expected)",
            len(df) == EXPECTED_ROWS,
            observed=len(df),
            expected=EXPECTED_ROWS,
        )
    )

    # Per-column null counts + missing percentage (lecture included that: flag >20%)
    for col in df.columns:
        null_count = int(df[col].isnull().sum())
        pct = round(null_count / len(df) * 100, 2)
        flag = " [CRITICAL >20%]" if pct > 20 else ""
        results.append(
            _check(
                f"No nulls in '{col}'{flag}",
                null_count == 0,
                observed=f"{null_count} nulls ({pct}%)",
                expected="0 nulls (0.00%)",
            )
        )

    return results


# Dimension 4 — Uniqueness
# "Are entities recorded only once with no duplicates?"
# ===========================================================================


def validate_uniqueness(df: pd.DataFrame) -> list:
    log.info("Validating Dimension 4: Uniqueness")
    results = []

    # Exact duplicates (all columns identical)
    dup_exact = int(df.duplicated().sum())
    results.append(
        _check(
            "No exact duplicate rows (all columns identical)",
            dup_exact == 0,
            observed=f"{dup_exact} exact duplicates",
            expected="0",
        )
    )

    # Subset duplicates: same demographic profile
    # (LIMIT_BAL + SEX + EDUCATION + MARRIAGE + AGE)
    # I made that because the dataset does not have a unique ID column, so we can check for duplicates based on a combination of demographic features
    key_cols = [
        c
        for c in ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"]
        if c in df.columns
    ]
    if key_cols:
        dup_key = int(df.duplicated(subset=key_cols, keep=False).sum())
        results.append(
            _check(
                f"Subset duplicate check on {key_cols} (informational)",
                True,  # always passes — informational only to inspect
                observed=f"{dup_key} rows share the same demographic profile",
                expected="expected to have some — informational only",
            )
        )

    # Borrower diversity (proxy for fuzzy uniqueness)
    # If LIMIT_BAL has fewer than 10 distinct values, the dataset has collapsed to a lookup table rather than individual borrower records
    if "LIMIT_BAL" in df.columns:
        n_unique = int(df["LIMIT_BAL"].nunique())
        results.append(
            _check(
                "LIMIT_BAL has >= 10 distinct values (borrower diversity / fuzzy uniqueness proxy)",
                n_unique >= 10,
                observed=f"{n_unique} unique values",
                expected=">= 10",
            )
        )

    return results


# Dimension 5 — Outliers Detection
# "Employ statistical methods to identify outliers and anomalous patterns"
# ===========================================================================


def validate_outliers(df: pd.DataFrame) -> list:
    log.info("Validating Dimension 5: Outliers")
    results = []

    num_cols = [c for c in NUMERIC_COLS if c in df.columns]
    if not num_cols:
        results.append(
            _check(
                "Outlier check",
                False,
                observed="no numeric columns found",
                expected="numeric columns present",
            )
        )
        return results

    # Z-Score (threshold ±3)
    log.info("  Outlier Method 1: Z-Score (threshold ±3)")
    z_summary = {}
    for col in num_cols:
        # I drop nulls to avoid skewing the z-score calculation
        col_data = df[col].dropna()

        # Calculate z-scores and count how many values exceed the threshold of 3 in absolute value
        z_scores = np.abs(stats.zscore(col_data))

        n_out = int((z_scores > 3).sum())

        pct_out = round(n_out / len(col_data) * 100, 2)

        z_summary[col] = (n_out, pct_out)

    # Summarize Z-score outliers across all numeric columns
    # v[0] -> number of outliers
    # v[1] -> percentage of outliers
    # I sum v[0] across all columns to get the total count of detected outliers
    total_z = sum(v[0] for v in z_summary.values())

    # Identify the top 5 columns with the highest number of outliers
    # x[0] -> column name
    # x[1][0] -> outlier count for that column
    # Sort in descending order by outlier count, then keep only the top 5
    top_z = sorted(z_summary.items(), key=lambda x: x[1][0], reverse=True)[:5]

    top_z_str = ", ".join(f"{c}: {v[0]} ({v[1]}%)" for c, v in top_z)
    results.append(
        _check(
            "Z-Score outliers (|z| > 3) across continuous columns — top 5 columns",
            True,  # informational to inspect
            observed=f"total flagged={total_z} | top cols: {top_z_str}",
            expected="informational — review extreme values before modelling",
        )
    )

    # IQR
    log.info("  Outlier Method 2: IQR")
    iqr_summary = {}
    for col in num_cols:
        # I drop nulls to avoid skewing the IQR calculation, and because I already check for nulls in the Completeness dimension
        col_data = df[col].dropna()

        q1, q3 = col_data.quantile(0.25), col_data.quantile(0.75)

        iqr = q3 - q1

        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr

        # I count how many values are outside the [lo, hi] range defined by the IQR method
        n_out = int(((col_data < lo) | (col_data > hi)).sum())

        pct_out = round(n_out / len(col_data) * 100, 2)

        # I store the count and percentage of outliers for each column in the iqr_summary dictionary for later summarization
        iqr_summary[col] = (n_out, pct_out, round(lo, 2), round(hi, 2))

    # Summarize IQR outliers across all numeric columns
    # v[0] -> number of outliers
    # v[1] -> percentage of outliers
    # I sum v[0] across all columns to get the total count of detected outliers by the IQR method
    total_iqr = sum(v[0] for v in iqr_summary.values())

    # Identify the top 5 columns with the highest number of outliers according to the IQR method
    # x[0] -> column name
    # x[1][0] -> outlier count for that column according to the IQR method
    # Sort in descending order by outlier count, then keep only the top 5
    top_iqr = sorted(iqr_summary.items(), key=lambda x: x[1][0], reverse=True)[:5]
    top_iqr_str = ", ".join(f"{c}: {v[0]} ({v[1]}%)" for c, v in top_iqr)
    results.append(
        _check(
            "IQR outliers (Q1-1.5*IQR, Q3+1.5*IQR) across continuous columns — top 5",
            True,  # always passes — informational only to inspect
            observed=f"total flagged={total_iqr} | top cols: {top_iqr_str}",
            expected="informational — skewed financial features expected to have IQR outliers",
        )
    )

    # Isolation Forest (contamination=0.05)
    log.info("  Outlier Method 3: Isolation Forest (contamination=0.05)")
    try:

        # I make median imputation to fill nulls in numeric columns before running Isolation Forest, because it cannot handle null values
        # I choose median imputation because it is more robust to outliers than mean imputation
        iso_data = df[num_cols].fillna(df[num_cols].median())

        # I used contamination=0.05 to flag approximately 5% of the data as anomalies
        # This is a common default setting because I don't have a specific business rule for the expected anomaly rate
        # n_jobs=-1 allows the algorithm to use all available CPU cores for faster processing on larger datasets
        # random_state=42 ensures reproducibility of the results by fixing the random seed
        clf = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)

        # fit_predict returns an array where -1 indicates an anomaly and 1 indicates a normal point
        labels = clf.fit_predict(iso_data)

        # I count the number of anomalies
        n_anomalies = int((labels == -1).sum())
        pct_anom = round(n_anomalies / len(df) * 100, 2)
        results.append(
            _check(
                f"Isolation Forest anomaly detection (contamination=0.05, {len(num_cols)} features)",
                True,  # always passes — informational only to inspect
                observed=f"{n_anomalies} anomalies detected ({pct_anom}%)",
                expected="~5% flagged by design — review multivariate outliers before modelling",
            )
        )
    except Exception as exc:
        results.append(
            _check(
                "Isolation Forest anomaly detection",
                False,  # failed to run
                observed=f"ERROR: {exc}",
                expected="should run without error",
            )
        )

    # Outlier quality metric
    # Using IQR as the reference method for the quality score because it is more robust to skewed distributions than Z-Score, and Isolation Forest is more complex and less interpretable for a simple quality metric
    total_iqr_flagged = sum(v[0] for v in iqr_summary.values())
    total_values = sum(len(df[c].dropna()) for c in num_cols)
    pct_clean = (
        round((1 - total_iqr_flagged / total_values) * 100, 2)
        if total_values > 0
        else 0
    )
    results.append(
        _check(
            "Outlier quality metric: % non-outlier values (IQR method, all numeric cols)",
            pct_clean >= 80,
            observed=f"{pct_clean}% clean values",
            expected=">= 80% non-outlier",
        )
    )

    return results


# Dimension 6 — Timeliness
# "Validating time-based data properties."
# ===========================================================================


def validate_timeliness(df: pd.DataFrame) -> list:
    log.info("Validating Dimension 6: Timeliness")
    results = []

    # Values consistent with post-2010 or post-2022 data are wrong
    # I check that there is no values by accident from the post-2010 or post-2022 period
    # I made that to indicate a timeliness issue if the data acquisition step fetched the wrong time period from the source
    timeliness_checks = [
        (
            "avg_macro_rate",
            5.0,
            "Discount Rate < 5.0% (pre-2010 CBC — ceiling catches wrong-period fetch)",
        ),
        (
            "avg_macro_TAIEX",
            10_000,
            "TAIEX < 10,000 (pre-2010 TWSE — ceiling catches post-2008 bull market data)",
        ),
        (
            "avg_macro_unemp",
            6.0,
            "Unemployment < 6.0% (pre-GFC ceiling — confirms 2005 DGBAS period)",
        ),
        (
            "avg_macro_CPI",
            200.0,
            "CPI < 200.0 (pre-2022 ceiling — catches 2022 inflation-era FRED data)",
        ),
    ]
    for col, ceiling, label in timeliness_checks:
        if col not in df.columns:
            continue

        # Count bad rows
        bad = int((df[col] >= ceiling).sum())

        # Calculate their mean
        mean = round(float(df[col].mean()), 4)
        results.append(
            _check(
                label,
                bad == 0,
                observed=f"mean={mean}, {bad} rows at or above ceiling",
                expected=f"all values < {ceiling}",
            )
        )

    # Expected frequency: macro columns should have exactly 6 distinct months
    # embedded in the panel (Apr-Sep 2005). Because we aggregate to borrower level, we verify plausibility via constant value check instead
    # because if there is much distinct values then no way that all borrowers share the same 6-month window, which indicates a timeliness issue
    for col in [
        "avg_macro_CPI",
        "avg_macro_GDP",
        "avg_macro_TAIEX",
        "avg_macro_rate",
        "avg_macro_unemp",
    ]:
        if col not in df.columns:
            continue
        n_distinct = int(df[col].nunique())
        # After borrower-level averaging these ARE constant (all borrowers share
        # the same 6-month window), so we expect very few distinct values (1-6)
        results.append(
            _check(
                f"'{col}' has <= 6 distinct values (confirms 6-month window, no gap/duplicate months)",
                n_distinct <= 6,
                observed=f"{n_distinct} distinct values",
                expected="<= 6 (one per observation month Apr-Sep 2005)",
            )
        )

    return results


# Dimension 7 — Distribution Profile
# "Examines how values are distributed within fields."
# ===========================================================================


def validate_distribution(df: pd.DataFrame) -> list:
    log.info("Validating Dimension 7: Distribution Profile")
    results = []

    num_cols = [c for c in NUMERIC_COLS if c in df.columns]

    # Basic profile per column
    # avoid skew/kurtosis noise from nulls because we already check for nulls in completeness dimension
    for col in num_cols:
        s = df[col].dropna()
        if len(s) == 0:
            continue
        col_skew = round(float(skew(s)), 3)
        col_kurt = round(float(kurtosis(s)), 3)  # excess kurtosis (0 = normal)

        # Interpret skewness as symmetric, moderately skewed, or highly skewed based on threshold
        if abs(col_skew) < 0.5:
            skew_label = "symmetric (|skew|<0.5)"
        elif col_skew > 0:
            skew_label = f"positive skew ({col_skew})"
        else:
            skew_label = f"negative skew ({col_skew})"

        # Interpret kurtosis as leptokurtic, platykurtic, or mesokurtic based on excess kurtosis thresholds
        if col_kurt > 1:
            kurt_label = f"leptokurtic (excess={col_kurt})"
        elif col_kurt < -1:
            kurt_label = f"platykurtic (excess={col_kurt})"
        else:
            kurt_label = f"mesokurtic (excess={col_kurt})"

        # I also include the 25th and 75th percentiles to give more insight into the distribution shape, especially for skewed financial features
        q25, q50, q75 = s.quantile([0.25, 0.50, 0.75])

        results.append(
            _check(
                f"'{col}' distribution profile",
                True,  # always passes — informational only to inspect
                observed=(
                    f"min={s.min():.2f}, max={s.max():.2f}, "
                    f"mean={s.mean():.2f}, median={q50:.2f}, std={s.std():.2f} | "
                    f"Q25={q25:.2f}, Q75={q75:.2f} | "
                    f"skew={skew_label} | kurt={kurt_label} | "
                    f"cardinality={s.nunique()}"
                ),
                expected="informational — use skew/kurt to decide transformations in preprocessing",
            )
        )

    # KS test: does each numeric column follow a normal distribution?
    # "If p-value < 0.05, the data does not follow the assumed distribution"
    log.info("  Distribution: KS test against normal distribution")
    for col in num_cols:
        s = df[col].dropna()
        if len(s) < 50:
            continue

        # I use the sample mean and std as parameters for the normal distribution in the KS test, which is a common approach to test for normality when parameters are estimated from the data
        ks_stat, p_val = kstest(s, "norm", args=(float(s.mean()), float(s.std())))
        ks_stat = round(ks_stat, 4)
        p_val = round(p_val, 4)

        # If the p-value is greater than or equal to 0.05, we fail to reject the null hypothesis that the data follows a normal distribution
        # so I can say that the data is "normal-like" in that case
        is_normal = p_val >= 0.05
        results.append(
            _check(
                f"KS test — '{col}' vs normal distribution (p >= 0.05 -> normal-like)",
                is_normal,
                observed=f"KS stat={ks_stat}, p={p_val}",
                expected="p >= 0.05 for normal-like; p < 0.05 -> non-normal -> consider log/power transform",
            )
        )

    # Class distribution (target variable)
    if "default payment next month" in df.columns:
        rate = float(df["default payment next month"].mean())
        results.append(
            _check(
                f"Default rate in [{DEFAULT_RATE_MIN:.0%}, {DEFAULT_RATE_MAX:.0%}] (published Yeh 2009: ~22.1%)",
                DEFAULT_RATE_MIN <= rate <= DEFAULT_RATE_MAX,
                observed=f"{rate:.4f} ({rate:.1%})",
                expected=f"[{DEFAULT_RATE_MIN:.2f}, {DEFAULT_RATE_MAX:.2f}]",
            )
        )
        minority_pct = rate
        results.append(
            _check(
                "Minority class (defaulters) > 10% — no extreme imbalance",
                minority_pct > 0.10,
                observed=f"defaulters={int(df['default payment next month'].sum())} ({minority_pct:.1%})",
                expected="> 10%",
            )
        )

    # Distribution checks from Yeh (2009) published facts
    # I want to know the mean of ages based on the published paper to check if the data distribution is consistent with the original source
    if "AGE" in df.columns:
        mean_age = float(df["AGE"].mean())
        results.append(
            _check(
                "Mean AGE in [30, 41] (published Yeh 2009: ~35.5)",
                30 <= mean_age <= 41,
                observed=round(mean_age, 2),
                expected="[30, 41]",
            )
        )

    # I want to check the mean of credit limits based on the published paper to see if the data distribution is consistent with the original source
    if "LIMIT_BAL" in df.columns:
        mean_limit = float(df["LIMIT_BAL"].mean())
        results.append(
            _check(
                "Mean LIMIT_BAL in [100K, 250K NTD] (published Yeh 2009: ~167K)",
                100_000 <= mean_limit <= 250_000,
                observed=round(mean_limit, 0),
                expected="[100000, 250000]",
            )
        )

    # PAY_0 and PAY_2 should have a majority of on-time payments (value <= 0) because the dataset is from a stable economic period with relatively low delinquency rates
    # so I expect at least 50% of payments to be on time in those columns
    for col in ["PAY_0", "PAY_2"]:
        if col not in df.columns:
            continue
        pct_on_time = float((df[col] <= 0).mean())
        results.append(
            _check(
                f"'{col}': >= 50% on-time (value <= 0) — majority of borrowers pay on time",
                pct_on_time >= 0.50,
                observed=f"{pct_on_time:.2%} on time",
                expected=">= 50%",
            )
        )

    return results


# Dimension 8 — Relationships Profile
# "Examines correlations and dependencies between fields."
# ===========================================================================


def validate_relationships(df: pd.DataFrame) -> list:
    log.info("Validating Dimension 8: Relationships Profile")
    results = []

    # We focus on numeric columns for correlation analysis, excluding the binary target and categorical features
    num_cols = [
        c
        for c in df.select_dtypes(include="number").columns
        if c != "default payment next month"
    ]

    # If fewer than 2 numeric columns are present, we cannot perform correlation analysis
    if len(num_cols) < 2:
        results.append(
            _check(
                "Relationship analysis",
                False,
                observed="fewer than 2 numeric columns",
                expected=">= 2",
            )
        )
        return results

    # Pearson vs Spearman comparison
    log.info("  Relationships: Pearson vs Spearman comparison")
    high_nonlinear = []
    sample_pairs = [
        ("LIMIT_BAL", "avg_bill"),
        ("LIMIT_BAL", "avg_payment"),
        ("avg_bill", "avg_payment"),
        ("AGE", "LIMIT_BAL"),
        ("total_delinquency", "max_delinquency"),
    ]
    for c1, c2 in sample_pairs:
        if c1 not in df.columns or c2 not in df.columns:
            continue

        # I calculate both Pearson and Spearman correlation coefficients for the pair of columns
        pearson = round(float(df[c1].corr(df[c2], method="pearson")), 4)
        spearman = round(float(df[c1].corr(df[c2], method="spearman")), 4)

        # I compare the absolute difference between Pearson and Spearman correlations to identify potential non-linear relationships
        diff = abs(pearson - spearman)
        nonlinear_flag = diff > 0.1
        if nonlinear_flag:
            high_nonlinear.append(f"{c1}↔{c2} (Δ={diff:.3f})")
        results.append(
            _check(
                f"Pearson vs Spearman: '{c1}' ↔ '{c2}' (Δ > 0.1 -> non-linear relationship)",
                not nonlinear_flag,
                observed=f"Pearson={pearson}, Spearman={spearman}, Δ={diff:.3f}",
                expected="Δ <= 0.1 (linear); Δ > 0.1 signals non-linear -> use Spearman",
            )
        )

    # High multicollinearity detection (Pearson |r| > 0.9)
    log.info("  Relationships: multicollinearity check (|Pearson| > 0.9)")

    # I cap for speed as correlation is O(n^2) so I take the first 20 numeric columns
    sub_cols = [c for c in num_cols if c in df.columns][:20]
    if len(sub_cols) >= 2:
        pearson_matrix = df[sub_cols].corr(method="pearson")
        high_pairs = []
        for i in range(len(sub_cols)):
            for j in range(i + 1, len(sub_cols)):
                r = abs(pearson_matrix.iloc[i, j])
                if r > 0.9:
                    high_pairs.append(f"{sub_cols[i]}↔{sub_cols[j]} (r={r:.3f})")
        results.append(
            _check(
                "No highly collinear feature pairs (|Pearson r| > 0.9) — destabilises models",
                len(high_pairs) == 0,
                observed=(
                    f"{len(high_pairs)} pairs: {high_pairs[:5]}"
                    if high_pairs
                    else "none"
                ),
                expected="0 pairs with |r| > 0.9",
            )
        )

    # Feature-to-target correlations
    # Expected: PAY_STATUS columns should correlate positively with default
    if "default payment next month" in df.columns:
        log.info("  Relationships: feature-to-target correlations")
        target = df["default payment next month"]
        expected_positive = ["PAY_0", "PAY_2", "total_delinquency", "max_delinquency"]
        for col in expected_positive:
            if col not in df.columns:
                continue
            r_p = round(float(df[col].corr(target, method="pearson")), 4)
            r_s = round(float(df[col].corr(target, method="spearman")), 4)
            results.append(
                _check(
                    f"'{col}' positively correlated with default target (expected — delinquency predicts default)",
                    r_p > 0,
                    observed=f"Pearson={r_p}, Spearman={r_s}",
                    expected="Pearson > 0 (delinquency -> higher default risk)",
                )
            )

        # LIMIT_BAL expected to be negatively correlated (higher limit -> lower risk)
        if "LIMIT_BAL" in df.columns:
            r = round(float(df["LIMIT_BAL"].corr(target, method="pearson")), 4)
            results.append(
                _check(
                    "LIMIT_BAL negatively correlated with default (higher credit limit -> lower default risk)",
                    r < 0,
                    observed=f"Pearson={r}",
                    expected="Pearson < 0",
                )
            )

    return results


# Integrity (structural completeness — "all columns present, no merge fan-out")
# ===========================================================================


def validate_integrity(df: pd.DataFrame) -> list:
    log.info("Validating Integrity (structural completeness)")
    results = []

    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    results.append(
        _check(
            "All expected columns present",
            len(missing_cols) == 0,
            observed=f"missing: {missing_cols}" if missing_cols else "none missing",
            expected=f"{len(EXPECTED_COLUMNS)} columns",
        )
    )

    n_cols = len(df.columns)
    col_lo = len(EXPECTED_COLUMNS)
    col_hi = len(EXPECTED_COLUMNS) + 5
    results.append(
        _check(
            f"Column count in [{col_lo}, {col_hi}] (allows up to +5 extra cols)",
            col_lo <= n_cols <= col_hi,
            observed=n_cols,
            expected=f"[{col_lo}, {col_hi}]",
        )
    )

    results.append(
        _check(
            "Row count = 30,000 (no merge fan-out)",
            len(df) == EXPECTED_ROWS,
            observed=len(df),
            expected=EXPECTED_ROWS,
        )
    )

    if "default payment next month" in df.columns:
        # Target column should be binary (0 or 1)
        non_binary = int((~df["default payment next month"].isin(TARGET_VALUES)).sum())
        results.append(
            _check(
                "Target column is binary {0, 1}",
                non_binary == 0,
                observed=f"{non_binary} non-binary values",
                expected="0",
            )
        )

    # All columns numeric (no object dtype creeping in from merge)
    object_cols = [c for c in df.columns if df[c].dtype == object]
    results.append(
        _check(
            "All columns are numeric (no object dtype from merge artefacts)",
            len(object_cols) == 0,
            observed=f"object cols: {object_cols}" if object_cols else "none",
            expected="none",
        )
    )

    return results


# Data Quality Metrics — Quantification table
# ===========================================================================


def compute_quality_metrics(df: pd.DataFrame, results: dict) -> dict:
    metrics = {}

    # Completeness: (non-null values / total values) x 100%
    total_vals = df.size

    # I count the total number of non-null values across the entire DataFrame by summing the non-null counts for each column and then summing those counts together
    non_null = int(df.notnull().sum().sum())

    metrics["Completeness"] = {
        "formula": "(non-null values / total values) x 100%",
        "value": round(non_null / total_vals * 100, 4),
        "unit": "%",
    }

    # Uniqueness: (unique rows / total rows) x 100%
    n_unique = len(df) - int(df.duplicated().sum())
    metrics["Uniqueness"] = {
        "formula": "(unique records / total records) x 100%",
        "value": round(n_unique / len(df) * 100, 4),
        "unit": "%",
    }

    # Accuracy: checks passed in accuracy dimension / total accuracy checks
    acc_checks = results.get("1_Accuracy", [])
    acc_passed = sum(1 for c in acc_checks if c["success"])

    metrics["Accuracy"] = {
        "formula": "(passed accuracy checks / total accuracy checks) x 100%",
        "value": round(acc_passed / len(acc_checks) * 100, 2) if acc_checks else 0,
        "unit": "%",
    }

    # Consistency: checks passed in consistency dimension
    con_checks = results.get("2_Consistency", [])
    con_passed = sum(1 for c in con_checks if c["success"])

    metrics["Consistency"] = {
        "formula": "(conforming records checks / total consistency checks) x 100%",
        "value": round(con_passed / len(con_checks) * 100, 2) if con_checks else 0,
        "unit": "%",
    }

    # Outliers: % non-outlier values (IQR method across numeric columns)
    num_cols = [c for c in NUMERIC_COLS if c in df.columns]
    if num_cols:
        total_v, flagged = 0, 0
        for col in num_cols:
            s = df[col].dropna()
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            flagged += int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
            total_v += len(s)

        metrics["Outliers"] = {
            "formula": "(non-outlier values / total values) x 100%  [IQR method]",
            "value": round((1 - flagged / total_v) * 100, 4) if total_v > 0 else 0,
            "unit": "%",
        }

    # Timeliness: checks passed / total
    tim_checks = results.get("6_Timeliness", [])
    tim_passed = sum(1 for c in tim_checks if c["success"])

    metrics["Timeliness"] = {
        "formula": "mean time from event to availability -> % timeliness checks passed",
        "value": round(tim_passed / len(tim_checks) * 100, 2) if tim_checks else 0,
        "unit": "%",
    }

    # Distribution: mean KS statistic across numeric columns
    ks_stats = []
    for col in num_cols:
        s = df[col].dropna()
        if len(s) >= 50:
            ks_stat, _ = kstest(s, "norm", args=(float(s.mean()), float(s.std())))
            ks_stats.append(ks_stat)

    metrics["Distribution"] = {
        "formula": "mean KS statistic across numeric columns (0=identical to normal, 1=completely different)",
        "value": round(float(np.mean(ks_stats)), 4) if ks_stats else None,
        "unit": "KS statistic",
    }

    # Relationships: mean absolute Spearman r on key feature pairs
    target_pairs = [
        ("PAY_0", "default payment next month"),
        ("total_delinquency", "default payment next month"),
        ("LIMIT_BAL", "default payment next month"),
    ]
    spearman_vals = []
    for c1, c2 in target_pairs:
        if c1 in df.columns and c2 in df.columns:
            r = df[c1].corr(df[c2], method="spearman")
            spearman_vals.append(abs(r))

    metrics["Relationships"] = {
        "formula": "mean |Spearman r| for key feature-to-target pairs",
        "value": round(float(np.mean(spearman_vals)), 4) if spearman_vals else None,
        "unit": "Spearman |r|",
    }

    return metrics


# Report generator
# ===========================================================================

_DIMENSION_META = {
    "1_Accuracy": (
        "Accuracy",
        "Does data correctly represent reality? Cross-checked against 2005 Taiwan benchmarks and business rules (Age > 0, credit limit positive, email format, etc.).",
    ),
    "2_Consistency": (
        "Consistency",
        "Is data uniform across systems and time periods? Validates logical cross-column relationships (e.g. delinquency sum vs individual PAY_STATUS, payment floors).",
    ),
    "3_Completeness": (
        "Completeness",
        "Is all required data present? Checks missing value counts and percentages per column (flags >20% missing), and total row count vs expected.",
    ),
    "4_Uniqueness": (
        "Uniqueness",
        "Are entities recorded only once? Covers exact duplicates, subset (key-field) duplicates, and a borrower-diversity proxy for fuzzy uniqueness.",
    ),
    "5_Outliers": (
        "Outliers Detection",
        "Statistical methods to identify anomalous patterns: Z-Score (threshold ±3), IQR (Q1-1.5*IQR, Q3+1.5*IQR), and Isolation Forest (multivariate, contamination=0.05).",
    ),
    "6_Timeliness": (
        "Timeliness",
        "Validates time-based data properties: macro data corresponds to Apr-Sep 2005 (not 2010+ or 2022+), expected 6-month frequency window, no gaps.",
    ),
    "7_Distribution": (
        "Distribution Profile",
        "Examines how values are distributed: basic stats, skewness (positive/negative/symmetric), kurtosis (leptokurtic/mesokurtic/platykurtic), KS test vs normal.",
    ),
    "8_Relationships": (
        "Relationships Profile",
        "Examines correlations and dependencies: Pearson vs Spearman comparison (Δ > 0.1 -> non-linear), multicollinearity detection (|r| > 0.9), feature-to-target correlations.",
    ),
    "9_Integrity": (
        "Integrity (Structural)",
        "Project-rubric structural checks: all expected columns present, column count in range, row count exactly 30,000 (no merge fan-out), target column binary.",
    ),
}


def _generate_markdown_report(results: dict, metrics: dict, output_path: str) -> None:
    total_checks = sum(len(v) for v in results.values())
    total_passed = sum(sum(1 for c in v if c["success"]) for v in results.values())
    total_failed = total_checks - total_passed

    lines = [
        "# Data Validation Report",
        f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "**Source:** taiwan_merged.csv",
        f"\n**Total checks:** {total_checks} | **Passed:** {total_passed} | **Failed:** {total_failed}",
        "\n---\n",
        "## Data Quality Metrics Summary\n",
        "*(Quantification table from ADS Lecture 03 — Data Validation)*\n",
        "| Dimension | Formula | Value | Unit |",
        "|-----------|---------|-------|------|",
    ]

    # I iterate over the computed quality metrics for each dimension and add a row to the markdown table with the dimension name, formula, computed value, and unit
    # If a metric value is None, I display "N/A" in the table
    for dim, m in metrics.items():
        val = m["value"] if m["value"] is not None else "N/A"
        lines.append(f"| {dim} | {m['formula']} | {val} | {m['unit']} |")

    lines.append("\n---\n")

    # Detailed results per dimension
    for key in sorted(results.keys()):
        # I retrieve the list of checks for the current dimension and count how many passed and failed
        checks = results[key]

        # I get the dimension name and description from the _DIMENSION_META dictionary using the key, and if the key is not found, I default to using the key itself as the name and an empty string as the description
        dim_name, dim_desc = _DIMENSION_META.get(key, (key, ""))

        # Count passed and failed checks for this dimension
        passed = sum(1 for c in checks if c["success"])
        failed = len(checks) - passed

        lines.append(f"## {key[0:2].strip('_')}. {dim_name}")
        lines.append(f"\n*{dim_desc}*\n")
        lines.append(
            f"**Checks:** {len(checks)} | **Passed:** {passed} | **Failed:** {failed}\n"
        )
        lines.append("| Check | Status | Observed | Expected |")
        lines.append("|-------|--------|----------|----------|")
        for c in checks:
            status = "✅ PASS" if c["success"] else "❌ FAIL"
            lines.append(
                f"| {c['check']} | {status} | {c['observed']} | {c['expected']} |"
            )
        lines.append("")

    # Failures summary
    lines += ["\n---\n", "## ❌ Failed Checks — Action Required\n"]
    any_fail = False
    for key in sorted(results.keys()):
        dim_name = _DIMENSION_META.get(key, (key,))[0]
        failures = [c for c in results[key] if not c["success"]]
        if failures:
            any_fail = True
            lines.append(f"### {dim_name}")
            for c in failures:
                lines.append(
                    f"- **{c['check']}** — observed: {c['observed']} (expected: {c['expected']})"
                )
            lines.append("")
    if not any_fail:
        lines.append("✅ No failures detected. Dataset passes all 8 dimensions.")

    # Notes for transformation
    lines += ["\n---\n", "## Notes for Transformation Step\n"]
    notes = [
        (
            "avg_macro_CPI, avg_macro_GDP, avg_macro_TAIEX, avg_macro_rate, avg_macro_unemp are CONSTANT "
            "across all 30,000 borrowers (same 6-month window for everyone). Their values are accurate as "
            "confirmed above, but they carry zero variance for modelling. The transformation step must "
            "replace them with interaction features (e.g. CPI only during this borrower's delinquent months)."
        ),
        (
            "PAY_STATUS columns (PAY_0, PAY_2-PAY_6) use the raw UCI encoding (-2 to 8). The transformation "
            "step must recode these before feeding them to any model that assumes ordinal monotonicity. "
            "-2 (no consumption) and -1 (paid in full) both mean no delinquency but are numerically negative."
        ),
        (
            "EDUCATION and MARRIAGE contain undocumented codes (0, 5, 6 for EDUCATION; 0 for MARRIAGE). "
            "The transformation step must decide whether to remap or group them."
        ),
        (
            "Distribution profile reveals that BILL_AMT and PAY_AMT columns are likely highly right-skewed "
            "(positive skew, leptokurtic). The transformation step should consider log1p or power transforms."
        ),
        (
            "Isolation Forest detected ~5% multivariate anomalies. Review these borrowers before deciding "
            "whether to treat them as legitimate extreme cases or data errors."
        ),
    ]
    for note in notes:
        lines.append(f"- {note}\n")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log.info("Markdown report saved: %s", output_path)


def run_validation(
    merged_path: str = "data/taiwan_merged.csv",
    json_output: str = "data/validation_results.json",
    md_output: str = "data/validation_report.md",
) -> dict:
    log.info("Loading merged dataset: %s", merged_path)
    df = pd.read_csv(merged_path)
    log.info("Loaded: %s rows x %s cols", *df.shape)

    results = {
        "1_Accuracy": validate_accuracy(df),
        "2_Consistency": validate_consistency(df),
        "3_Completeness": validate_completeness(df),
        "4_Uniqueness": validate_uniqueness(df),
        "5_Outliers": validate_outliers(df),
        "6_Timeliness": validate_timeliness(df),
        "7_Distribution": validate_distribution(df),
        "8_Relationships": validate_relationships(df),
        "9_Integrity": validate_integrity(df),
    }

    total = sum(len(v) for v in results.values())
    passed = sum(sum(1 for c in v if c["success"]) for v in results.values())
    failed = total - passed

    log.info(
        "Validation complete: %d checks | %d passed | %d failed", total, passed, failed
    )

    metrics = compute_quality_metrics(df, results)

    # Save JSON
    Path(json_output).parent.mkdir(parents=True, exist_ok=True)
    with open(json_output, "w", encoding="utf-8") as f:
        json.dump({"results": results, "quality_metrics": metrics}, f, indent=2)
    log.info("JSON results saved: %s", json_output)

    _generate_markdown_report(results, metrics, md_output)

    if failed > 0:
        log.warning(
            "%d check(s) FAILED. Review %s before proceeding.", failed, md_output
        )
    else:
        log.info("All checks passed. Dataset is ready for transformation.")

    return results


if __name__ == "__main__":
    run_validation()

if __name__ != "__main__":
    try:
        if Path("data/taiwan_merged.csv").exists():
            run_validation()
    except Exception as e:
        log.warning(f"Auto-validation failed during import: {e}")
