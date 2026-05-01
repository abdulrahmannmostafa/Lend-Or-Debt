# Data Validation Report

**Generated:** 2026-05-02 01:15:42
**Source:** taiwan_merged.csv

**Total checks:** 121 | **Passed:** 94 | **Failed:** 27

---

## Data Quality Metrics Summary

*(Quantification table from ADS Lecture 03 — Data Validation)*

| Dimension | Formula | Value | Unit |
|-----------|---------|-------|------|
| Completeness | (non-null values / total values) x 100% | 100.0 | % |
| Uniqueness | (unique records / total records) x 100% | 99.8833 | % |
| Accuracy | (passed accuracy checks / total accuracy checks) x 100% | 100.0 | % |
| Consistency | (conforming records checks / total consistency checks) x 100% | 40.0 | % |
| Outliers | (non-outlier values / total values) x 100%  [IQR method] | 92.0479 | % |
| Timeliness | mean time from event to availability -> % timeliness checks passed | 100.0 | % |
| Distribution | mean KS statistic across numeric columns (0=identical to normal, 1=completely different) | 0.2798 | KS statistic |
| Relationships | mean |Spearman r| for key feature-to-target pairs | 0.2841 | Spearman |r| |

---

## 1. Accuracy

*Does data correctly represent reality? Cross-checked against 2005 Taiwan benchmarks and business rules (Age > 0, credit limit positive, email format, etc.).*

**Checks:** 7 | **Passed:** 7 | **Failed:** 0

| Check | Status | Observed | Expected |
|-------|--------|----------|----------|
| AGE in [18, 100] (business rule: credit card eligibility floor) | ✅ PASS | 0 out-of-range | 0 |
| LIMIT_BAL in (0, 1B NTD] (business rule: positive credit limit) | ✅ PASS | 0 out-of-range | 0 |
| CPI in 2005 Taiwan published range [100, 200] (DGBAS data) | ✅ PASS | mean=195.1333, 0 out-of-range rows | all values in [100, 200] |
| GDP index in 2005 FRED range [10000, 15000] | ✅ PASS | mean=13032.649, 0 out-of-range rows | all values in [10000.0, 15000.0] |
| TAIEX in 2005 TWSE range [5000, 7500] | ✅ PASS | mean=6071.1404, 0 out-of-range rows | all values in [5000.0, 7500.0] |
| Discount Rate in 2005 CBC range [1.5, 2.5] | ✅ PASS | mean=2.0625, 0 out-of-range rows | all values in [1.5, 2.5] |
| Unemployment in 2005 DGBAS range [3.0, 5.5] | ✅ PASS | mean=4.1, 0 out-of-range rows | all values in [3.0, 5.5] |

## 2. Consistency

*Is data uniform across systems and time periods? Validates logical cross-column relationships (e.g. delinquency sum vs individual PAY_STATUS, payment floors).*

**Checks:** 10 | **Passed:** 4 | **Failed:** 6

| Check | Status | Observed | Expected |
|-------|--------|----------|----------|
| Borrowers with total_delinquency=0 have no PAY_STATUS > 0 (logical cross-column) | ✅ PASS | 0 inconsistent rows | 0 |
| max_delinquency <= 8 (ceiling of PAY_STATUS scale) | ✅ PASS | 0 values above 8 | 0 |
| avg_payment >= 0 (payments are non-negative cash flows) | ✅ PASS | 0 negative values | 0 |
| avg_bill > -1,000,000 NTD (credit note floor) | ✅ PASS | 0 below floor | 0 |
| PAY_AMT1 / BILL_AMT1 <= 10x (extreme overpayment check) | ❌ FAIL | 461 rows with ratio > 10x | 0 |
| PAY_AMT2 / BILL_AMT2 <= 10x (extreme overpayment check) | ❌ FAIL | 512 rows with ratio > 10x | 0 |
| PAY_AMT3 / BILL_AMT3 <= 10x (extreme overpayment check) | ❌ FAIL | 475 rows with ratio > 10x | 0 |
| PAY_AMT4 / BILL_AMT4 <= 10x (extreme overpayment check) | ❌ FAIL | 404 rows with ratio > 10x | 0 |
| PAY_AMT5 / BILL_AMT5 <= 10x (extreme overpayment check) | ❌ FAIL | 421 rows with ratio > 10x | 0 |
| PAY_AMT6 / BILL_AMT6 <= 10x (extreme overpayment check) | ❌ FAIL | 496 rows with ratio > 10x | 0 |

## 3. Completeness

*Is all required data present? Checks missing value counts and percentages per column (flags >20% missing), and total row count vs expected.*

**Checks:** 34 | **Passed:** 34 | **Failed:** 0

| Check | Status | Observed | Expected |
|-------|--------|----------|----------|
| Row count = 30,000 (data size vs expected) | ✅ PASS | 30000 | 30000 |
| No nulls in 'LIMIT_BAL' | ✅ PASS | 0 nulls (0.0%) | 0 nulls (0.00%) |
| No nulls in 'SEX' | ✅ PASS | 0 nulls (0.0%) | 0 nulls (0.00%) |
| No nulls in 'EDUCATION' | ✅ PASS | 0 nulls (0.0%) | 0 nulls (0.00%) |
| No nulls in 'MARRIAGE' | ✅ PASS | 0 nulls (0.0%) | 0 nulls (0.00%) |
| No nulls in 'AGE' | ✅ PASS | 0 nulls (0.0%) | 0 nulls (0.00%) |
| No nulls in 'PAY_0' | ✅ PASS | 0 nulls (0.0%) | 0 nulls (0.00%) |
| No nulls in 'PAY_2' | ✅ PASS | 0 nulls (0.0%) | 0 nulls (0.00%) |
| No nulls in 'PAY_3' | ✅ PASS | 0 nulls (0.0%) | 0 nulls (0.00%) |
| No nulls in 'PAY_4' | ✅ PASS | 0 nulls (0.0%) | 0 nulls (0.00%) |
| No nulls in 'PAY_5' | ✅ PASS | 0 nulls (0.0%) | 0 nulls (0.00%) |
| No nulls in 'PAY_6' | ✅ PASS | 0 nulls (0.0%) | 0 nulls (0.00%) |
| No nulls in 'BILL_AMT1' | ✅ PASS | 0 nulls (0.0%) | 0 nulls (0.00%) |
| No nulls in 'BILL_AMT2' | ✅ PASS | 0 nulls (0.0%) | 0 nulls (0.00%) |
| No nulls in 'BILL_AMT3' | ✅ PASS | 0 nulls (0.0%) | 0 nulls (0.00%) |
| No nulls in 'BILL_AMT4' | ✅ PASS | 0 nulls (0.0%) | 0 nulls (0.00%) |
| No nulls in 'BILL_AMT5' | ✅ PASS | 0 nulls (0.0%) | 0 nulls (0.00%) |
| No nulls in 'BILL_AMT6' | ✅ PASS | 0 nulls (0.0%) | 0 nulls (0.00%) |
| No nulls in 'PAY_AMT1' | ✅ PASS | 0 nulls (0.0%) | 0 nulls (0.00%) |
| No nulls in 'PAY_AMT2' | ✅ PASS | 0 nulls (0.0%) | 0 nulls (0.00%) |
| No nulls in 'PAY_AMT3' | ✅ PASS | 0 nulls (0.0%) | 0 nulls (0.00%) |
| No nulls in 'PAY_AMT4' | ✅ PASS | 0 nulls (0.0%) | 0 nulls (0.00%) |
| No nulls in 'PAY_AMT5' | ✅ PASS | 0 nulls (0.0%) | 0 nulls (0.00%) |
| No nulls in 'PAY_AMT6' | ✅ PASS | 0 nulls (0.0%) | 0 nulls (0.00%) |
| No nulls in 'default payment next month' | ✅ PASS | 0 nulls (0.0%) | 0 nulls (0.00%) |
| No nulls in 'avg_bill' | ✅ PASS | 0 nulls (0.0%) | 0 nulls (0.00%) |
| No nulls in 'avg_payment' | ✅ PASS | 0 nulls (0.0%) | 0 nulls (0.00%) |
| No nulls in 'max_delinquency' | ✅ PASS | 0 nulls (0.0%) | 0 nulls (0.00%) |
| No nulls in 'total_delinquency' | ✅ PASS | 0 nulls (0.0%) | 0 nulls (0.00%) |
| No nulls in 'avg_macro_CPI' | ✅ PASS | 0 nulls (0.0%) | 0 nulls (0.00%) |
| No nulls in 'avg_macro_GDP' | ✅ PASS | 0 nulls (0.0%) | 0 nulls (0.00%) |
| No nulls in 'avg_macro_TAIEX' | ✅ PASS | 0 nulls (0.0%) | 0 nulls (0.00%) |
| No nulls in 'avg_macro_rate' | ✅ PASS | 0 nulls (0.0%) | 0 nulls (0.00%) |
| No nulls in 'avg_macro_unemp' | ✅ PASS | 0 nulls (0.0%) | 0 nulls (0.00%) |

## 4. Uniqueness

*Are entities recorded only once? Covers exact duplicates, subset (key-field) duplicates, and a borrower-diversity proxy for fuzzy uniqueness.*

**Checks:** 3 | **Passed:** 2 | **Failed:** 1

| Check | Status | Observed | Expected |
|-------|--------|----------|----------|
| No exact duplicate rows (all columns identical) | ❌ FAIL | 35 exact duplicates | 0 |
| Subset duplicate check on ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE'] (informational) | ✅ PASS | 24870 rows share the same demographic profile | expected to have some — informational only |
| LIMIT_BAL has >= 10 distinct values (borrower diversity / fuzzy uniqueness proxy) | ✅ PASS | 81 unique values | >= 10 |

## 5. Outliers Detection

*Statistical methods to identify anomalous patterns: Z-Score (threshold ±3), IQR (Q1-1.5*IQR, Q3+1.5*IQR), and Isolation Forest (multivariate, contamination=0.05).*

**Checks:** 4 | **Passed:** 4 | **Failed:** 0

| Check | Status | Observed | Expected |
|-------|--------|----------|----------|
| Z-Score outliers (|z| > 3) across continuous columns — top 5 columns | ✅ PASS | total flagged=7747 | top cols: BILL_AMT1: 686 (2.29%), BILL_AMT4: 680 (2.27%), BILL_AMT2: 670 (2.23%), BILL_AMT3: 661 (2.2%), avg_bill: 660 (2.2%) | informational — review extreme values before modelling |
| IQR outliers (Q1-1.5*IQR, Q3+1.5*IQR) across continuous columns — top 5 | ✅ PASS | total flagged=38170 | top cols: PAY_AMT4: 2994 (9.98%), PAY_AMT6: 2958 (9.86%), PAY_AMT5: 2945 (9.82%), avg_payment: 2898 (9.66%), PAY_AMT1: 2745 (9.15%) | informational — skewed financial features expected to have IQR outliers |
| Isolation Forest anomaly detection (contamination=0.05, 16 features) | ✅ PASS | 1500 anomalies detected (5.0%) | ~5% flagged by design — review multivariate outliers before modelling |
| Outlier quality metric: % non-outlier values (IQR method, all numeric cols) | ✅ PASS | 92.05% clean values | >= 80% non-outlier |

## 6. Timeliness

*Validates time-based data properties: macro data corresponds to Apr-Sep 2005 (not 2010+ or 2022+), expected 6-month frequency window, no gaps.*

**Checks:** 9 | **Passed:** 9 | **Failed:** 0

| Check | Status | Observed | Expected |
|-------|--------|----------|----------|
| Discount Rate < 5.0% (pre-2010 CBC — ceiling catches wrong-period fetch) | ✅ PASS | mean=2.0625, 0 rows at or above ceiling | all values < 5.0 |
| TAIEX < 10,000 (pre-2010 TWSE — ceiling catches post-2008 bull market data) | ✅ PASS | mean=6071.1404, 0 rows at or above ceiling | all values < 10000 |
| Unemployment < 6.0% (pre-GFC ceiling — confirms 2005 DGBAS period) | ✅ PASS | mean=4.1, 0 rows at or above ceiling | all values < 6.0 |
| CPI < 200.0 (pre-2022 ceiling — catches 2022 inflation-era FRED data) | ✅ PASS | mean=195.1333, 0 rows at or above ceiling | all values < 200.0 |
| 'avg_macro_CPI' has <= 6 distinct values (confirms 6-month window, no gap/duplicate months) | ✅ PASS | 1 distinct values | <= 6 (one per observation month Apr-Sep 2005) |
| 'avg_macro_GDP' has <= 6 distinct values (confirms 6-month window, no gap/duplicate months) | ✅ PASS | 1 distinct values | <= 6 (one per observation month Apr-Sep 2005) |
| 'avg_macro_TAIEX' has <= 6 distinct values (confirms 6-month window, no gap/duplicate months) | ✅ PASS | 1 distinct values | <= 6 (one per observation month Apr-Sep 2005) |
| 'avg_macro_rate' has <= 6 distinct values (confirms 6-month window, no gap/duplicate months) | ✅ PASS | 1 distinct values | <= 6 (one per observation month Apr-Sep 2005) |
| 'avg_macro_unemp' has <= 6 distinct values (confirms 6-month window, no gap/duplicate months) | ✅ PASS | 1 distinct values | <= 6 (one per observation month Apr-Sep 2005) |

## 7. Distribution Profile

*Examines how values are distributed: basic stats, skewness (positive/negative/symmetric), kurtosis (leptokurtic/mesokurtic/platykurtic), KS test vs normal.*

**Checks:** 38 | **Passed:** 22 | **Failed:** 16

| Check | Status | Observed | Expected |
|-------|--------|----------|----------|
| 'LIMIT_BAL' distribution profile | ✅ PASS | min=10000.00, max=1000000.00, mean=167484.32, median=140000.00, std=129747.66 | Q25=50000.00, Q75=240000.00 | skew=positive skew (0.993) | kurt=mesokurtic (excess=0.536) | cardinality=81 | informational — use skew/kurt to decide transformations in preprocessing |
| 'AGE' distribution profile | ✅ PASS | min=21.00, max=79.00, mean=35.49, median=34.00, std=9.22 | Q25=28.00, Q75=41.00 | skew=positive skew (0.732) | kurt=mesokurtic (excess=0.044) | cardinality=56 | informational — use skew/kurt to decide transformations in preprocessing |
| 'BILL_AMT1' distribution profile | ✅ PASS | min=-165580.00, max=964511.00, mean=51223.33, median=22381.50, std=73635.86 | Q25=3558.75, Q75=67091.00 | skew=positive skew (2.664) | kurt=leptokurtic (excess=9.804) | cardinality=22723 | informational — use skew/kurt to decide transformations in preprocessing |
| 'BILL_AMT2' distribution profile | ✅ PASS | min=-69777.00, max=983931.00, mean=49179.08, median=21200.00, std=71173.77 | Q25=2984.75, Q75=64006.25 | skew=positive skew (2.705) | kurt=leptokurtic (excess=10.301) | cardinality=22346 | informational — use skew/kurt to decide transformations in preprocessing |
| 'BILL_AMT3' distribution profile | ✅ PASS | min=-157264.00, max=1664089.00, mean=47013.15, median=20088.50, std=69349.39 | Q25=2666.25, Q75=60164.75 | skew=positive skew (3.088) | kurt=leptokurtic (excess=19.78) | cardinality=22026 | informational — use skew/kurt to decide transformations in preprocessing |
| 'BILL_AMT4' distribution profile | ✅ PASS | min=-170000.00, max=891586.00, mean=43262.95, median=19052.00, std=64332.86 | Q25=2326.75, Q75=54506.00 | skew=positive skew (2.822) | kurt=leptokurtic (excess=11.307) | cardinality=21548 | informational — use skew/kurt to decide transformations in preprocessing |
| 'BILL_AMT5' distribution profile | ✅ PASS | min=-81334.00, max=927171.00, mean=40311.40, median=18104.50, std=60797.16 | Q25=1763.00, Q75=50190.50 | skew=positive skew (2.876) | kurt=leptokurtic (excess=12.304) | cardinality=21010 | informational — use skew/kurt to decide transformations in preprocessing |
| 'BILL_AMT6' distribution profile | ✅ PASS | min=-339603.00, max=961664.00, mean=38871.76, median=17071.00, std=59554.11 | Q25=1256.00, Q75=49198.25 | skew=positive skew (2.847) | kurt=leptokurtic (excess=12.268) | cardinality=20604 | informational — use skew/kurt to decide transformations in preprocessing |
| 'PAY_AMT1' distribution profile | ✅ PASS | min=0.00, max=873552.00, mean=5663.58, median=2100.00, std=16563.28 | Q25=1000.00, Q75=5006.00 | skew=positive skew (14.668) | kurt=leptokurtic (excess=415.185) | cardinality=7943 | informational — use skew/kurt to decide transformations in preprocessing |
| 'PAY_AMT2' distribution profile | ✅ PASS | min=0.00, max=1684259.00, mean=5921.16, median=2009.00, std=23040.87 | Q25=833.00, Q75=5000.00 | skew=positive skew (30.452) | kurt=leptokurtic (excess=1641.358) | cardinality=7899 | informational — use skew/kurt to decide transformations in preprocessing |
| 'PAY_AMT3' distribution profile | ✅ PASS | min=0.00, max=896040.00, mean=5225.68, median=1800.00, std=17606.96 | Q25=390.00, Q75=4505.00 | skew=positive skew (17.216) | kurt=leptokurtic (excess=564.217) | cardinality=7518 | informational — use skew/kurt to decide transformations in preprocessing |
| 'PAY_AMT4' distribution profile | ✅ PASS | min=0.00, max=621000.00, mean=4826.08, median=1500.00, std=15666.16 | Q25=296.00, Q75=4013.25 | skew=positive skew (12.904) | kurt=leptokurtic (excess=277.287) | cardinality=6937 | informational — use skew/kurt to decide transformations in preprocessing |
| 'PAY_AMT5' distribution profile | ✅ PASS | min=0.00, max=426529.00, mean=4799.39, median=1500.00, std=15278.31 | Q25=252.50, Q75=4031.50 | skew=positive skew (11.127) | kurt=leptokurtic (excess=180.034) | cardinality=6897 | informational — use skew/kurt to decide transformations in preprocessing |
| 'PAY_AMT6' distribution profile | ✅ PASS | min=0.00, max=528666.00, mean=5215.50, median=1500.00, std=17777.47 | Q25=117.75, Q75=4000.00 | skew=positive skew (10.64) | kurt=leptokurtic (excess=167.133) | cardinality=6939 | informational — use skew/kurt to decide transformations in preprocessing |
| 'avg_bill' distribution profile | ✅ PASS | min=-56043.17, max=877313.83, mean=44976.95, median=21051.83, std=63260.72 | Q25=4781.33, Q75=57104.42 | skew=positive skew (2.735) | kurt=leptokurtic (excess=10.522) | cardinality=27370 | informational — use skew/kurt to decide transformations in preprocessing |
| 'avg_payment' distribution profile | ✅ PASS | min=0.00, max=627344.33, mean=5275.23, median=2397.17, std=10137.95 | Q25=1113.29, Q75=5583.92 | skew=positive skew (14.616) | kurt=leptokurtic (excess=607.655) | cardinality=19180 | informational — use skew/kurt to decide transformations in preprocessing |
| KS test — 'LIMIT_BAL' vs normal distribution (p >= 0.05 -> normal-like) | ❌ FAIL | KS stat=0.1151, p=0.0 | p >= 0.05 for normal-like; p < 0.05 -> non-normal -> consider log/power transform |
| KS test — 'AGE' vs normal distribution (p >= 0.05 -> normal-like) | ❌ FAIL | KS stat=0.0944, p=0.0 | p >= 0.05 for normal-like; p < 0.05 -> non-normal -> consider log/power transform |
| KS test — 'BILL_AMT1' vs normal distribution (p >= 0.05 -> normal-like) | ❌ FAIL | KS stat=0.2367, p=0.0 | p >= 0.05 for normal-like; p < 0.05 -> non-normal -> consider log/power transform |
| KS test — 'BILL_AMT2' vs normal distribution (p >= 0.05 -> normal-like) | ❌ FAIL | KS stat=0.2369, p=0.0 | p >= 0.05 for normal-like; p < 0.05 -> non-normal -> consider log/power transform |
| KS test — 'BILL_AMT3' vs normal distribution (p >= 0.05 -> normal-like) | ❌ FAIL | KS stat=0.24, p=0.0 | p >= 0.05 for normal-like; p < 0.05 -> non-normal -> consider log/power transform |
| KS test — 'BILL_AMT4' vs normal distribution (p >= 0.05 -> normal-like) | ❌ FAIL | KS stat=0.2408, p=0.0 | p >= 0.05 for normal-like; p < 0.05 -> non-normal -> consider log/power transform |
| KS test — 'BILL_AMT5' vs normal distribution (p >= 0.05 -> normal-like) | ❌ FAIL | KS stat=0.2435, p=0.0 | p >= 0.05 for normal-like; p < 0.05 -> non-normal -> consider log/power transform |
| KS test — 'BILL_AMT6' vs normal distribution (p >= 0.05 -> normal-like) | ❌ FAIL | KS stat=0.2458, p=0.0 | p >= 0.05 for normal-like; p < 0.05 -> non-normal -> consider log/power transform |
| KS test — 'PAY_AMT1' vs normal distribution (p >= 0.05 -> normal-like) | ❌ FAIL | KS stat=0.3662, p=0.0 | p >= 0.05 for normal-like; p < 0.05 -> non-normal -> consider log/power transform |
| KS test — 'PAY_AMT2' vs normal distribution (p >= 0.05 -> normal-like) | ❌ FAIL | KS stat=0.3986, p=0.0 | p >= 0.05 for normal-like; p < 0.05 -> non-normal -> consider log/power transform |
| KS test — 'PAY_AMT3' vs normal distribution (p >= 0.05 -> normal-like) | ❌ FAIL | KS stat=0.3833, p=0.0 | p >= 0.05 for normal-like; p < 0.05 -> non-normal -> consider log/power transform |
| KS test — 'PAY_AMT4' vs normal distribution (p >= 0.05 -> normal-like) | ❌ FAIL | KS stat=0.379, p=0.0 | p >= 0.05 for normal-like; p < 0.05 -> non-normal -> consider log/power transform |
| KS test — 'PAY_AMT5' vs normal distribution (p >= 0.05 -> normal-like) | ❌ FAIL | KS stat=0.3767, p=0.0 | p >= 0.05 for normal-like; p < 0.05 -> non-normal -> consider log/power transform |
| KS test — 'PAY_AMT6' vs normal distribution (p >= 0.05 -> normal-like) | ❌ FAIL | KS stat=0.3846, p=0.0 | p >= 0.05 for normal-like; p < 0.05 -> non-normal -> consider log/power transform |
| KS test — 'avg_bill' vs normal distribution (p >= 0.05 -> normal-like) | ❌ FAIL | KS stat=0.234, p=0.0 | p >= 0.05 for normal-like; p < 0.05 -> non-normal -> consider log/power transform |
| KS test — 'avg_payment' vs normal distribution (p >= 0.05 -> normal-like) | ❌ FAIL | KS stat=0.3014, p=0.0 | p >= 0.05 for normal-like; p < 0.05 -> non-normal -> consider log/power transform |
| Default rate in [20%, 25%] (published Yeh 2009: ~22.1%) | ✅ PASS | 0.2212 (22.1%) | [0.20, 0.25] |
| Minority class (defaulters) > 10% — no extreme imbalance | ✅ PASS | defaulters=6636 (22.1%) | > 10% |
| Mean AGE in [30, 41] (published Yeh 2009: ~35.5) | ✅ PASS | 35.49 | [30, 41] |
| Mean LIMIT_BAL in [100K, 250K NTD] (published Yeh 2009: ~167K) | ✅ PASS | 167484.0 | [100000, 250000] |
| 'PAY_0': >= 50% on-time (value <= 0) — majority of borrowers pay on time | ✅ PASS | 77.27% on time | >= 50% |
| 'PAY_2': >= 50% on-time (value <= 0) — majority of borrowers pay on time | ✅ PASS | 85.21% on time | >= 50% |

## 8. Relationships Profile

*Examines correlations and dependencies: Pearson vs Spearman comparison (Δ > 0.1 -> non-linear), multicollinearity detection (|r| > 0.9), feature-to-target correlations.*

**Checks:** 11 | **Passed:** 7 | **Failed:** 4

| Check | Status | Observed | Expected |
|-------|--------|----------|----------|
| Pearson vs Spearman: 'LIMIT_BAL' ↔ 'avg_bill' (Δ > 0.1 -> non-linear relationship) | ❌ FAIL | Pearson=0.302, Spearman=0.0919, Δ=0.210 | Δ <= 0.1 (linear); Δ > 0.1 signals non-linear -> use Spearman |
| Pearson vs Spearman: 'LIMIT_BAL' ↔ 'avg_payment' (Δ > 0.1 -> non-linear relationship) | ✅ PASS | Pearson=0.3527, Spearman=0.4018, Δ=0.049 | Δ <= 0.1 (linear); Δ > 0.1 signals non-linear -> use Spearman |
| Pearson vs Spearman: 'avg_bill' ↔ 'avg_payment' (Δ > 0.1 -> non-linear relationship) | ❌ FAIL | Pearson=0.3439, Spearman=0.5973, Δ=0.253 | Δ <= 0.1 (linear); Δ > 0.1 signals non-linear -> use Spearman |
| Pearson vs Spearman: 'AGE' ↔ 'LIMIT_BAL' (Δ > 0.1 -> non-linear relationship) | ✅ PASS | Pearson=0.1447, Spearman=0.1865, Δ=0.042 | Δ <= 0.1 (linear); Δ > 0.1 signals non-linear -> use Spearman |
| Pearson vs Spearman: 'total_delinquency' ↔ 'max_delinquency' (Δ > 0.1 -> non-linear relationship) | ❌ FAIL | Pearson=0.8493, Spearman=0.9883, Δ=0.139 | Δ <= 0.1 (linear); Δ > 0.1 signals non-linear -> use Spearman |
| No highly collinear feature pairs (|Pearson r| > 0.9) — destabilises models | ❌ FAIL | 6 pairs: ['BILL_AMT1↔BILL_AMT2 (r=0.951)', 'BILL_AMT2↔BILL_AMT3 (r=0.928)', 'BILL_AMT3↔BILL_AMT4 (r=0.924)', 'BILL_AMT4↔BILL_AMT5 (r=0.940)', 'BILL_AMT4↔BILL_AMT6 (r=0.901)'] | 0 pairs with |r| > 0.9 |
| 'PAY_0' positively correlated with default target (expected — delinquency predicts default) | ✅ PASS | Pearson=0.3248, Spearman=0.2922 | Pearson > 0 (delinquency -> higher default risk) |
| 'PAY_2' positively correlated with default target (expected — delinquency predicts default) | ✅ PASS | Pearson=0.2636, Spearman=0.2169 | Pearson > 0 (delinquency -> higher default risk) |
| 'total_delinquency' positively correlated with default target (expected — delinquency predicts default) | ✅ PASS | Pearson=0.376, Spearman=0.3904 | Pearson > 0 (delinquency -> higher default risk) |
| 'max_delinquency' positively correlated with default target (expected — delinquency predicts default) | ✅ PASS | Pearson=0.3704, Spearman=0.3728 | Pearson > 0 (delinquency -> higher default risk) |
| LIMIT_BAL negatively correlated with default (higher credit limit -> lower default risk) | ✅ PASS | Pearson=-0.1535 | Pearson < 0 |

## 9. Integrity (Structural)

*Project-rubric structural checks: all expected columns present, column count in range, row count exactly 30,000 (no merge fan-out), target column binary.*

**Checks:** 5 | **Passed:** 5 | **Failed:** 0

| Check | Status | Observed | Expected |
|-------|--------|----------|----------|
| All expected columns present | ✅ PASS | none missing | 33 columns |
| Column count in [33, 38] (allows up to +5 extra cols) | ✅ PASS | 33 | [33, 38] |
| Row count = 30,000 (no merge fan-out) | ✅ PASS | 30000 | 30000 |
| Target column is binary {0, 1} | ✅ PASS | 0 non-binary values | 0 |
| All columns are numeric (no object dtype from merge artefacts) | ✅ PASS | none | none |


---

## ❌ Failed Checks — Action Required

### Consistency
- **PAY_AMT1 / BILL_AMT1 <= 10x (extreme overpayment check)** — observed: 461 rows with ratio > 10x (expected: 0)
- **PAY_AMT2 / BILL_AMT2 <= 10x (extreme overpayment check)** — observed: 512 rows with ratio > 10x (expected: 0)
- **PAY_AMT3 / BILL_AMT3 <= 10x (extreme overpayment check)** — observed: 475 rows with ratio > 10x (expected: 0)
- **PAY_AMT4 / BILL_AMT4 <= 10x (extreme overpayment check)** — observed: 404 rows with ratio > 10x (expected: 0)
- **PAY_AMT5 / BILL_AMT5 <= 10x (extreme overpayment check)** — observed: 421 rows with ratio > 10x (expected: 0)
- **PAY_AMT6 / BILL_AMT6 <= 10x (extreme overpayment check)** — observed: 496 rows with ratio > 10x (expected: 0)

### Uniqueness
- **No exact duplicate rows (all columns identical)** — observed: 35 exact duplicates (expected: 0)

### Distribution Profile
- **KS test — 'LIMIT_BAL' vs normal distribution (p >= 0.05 -> normal-like)** — observed: KS stat=0.1151, p=0.0 (expected: p >= 0.05 for normal-like; p < 0.05 -> non-normal -> consider log/power transform)
- **KS test — 'AGE' vs normal distribution (p >= 0.05 -> normal-like)** — observed: KS stat=0.0944, p=0.0 (expected: p >= 0.05 for normal-like; p < 0.05 -> non-normal -> consider log/power transform)
- **KS test — 'BILL_AMT1' vs normal distribution (p >= 0.05 -> normal-like)** — observed: KS stat=0.2367, p=0.0 (expected: p >= 0.05 for normal-like; p < 0.05 -> non-normal -> consider log/power transform)
- **KS test — 'BILL_AMT2' vs normal distribution (p >= 0.05 -> normal-like)** — observed: KS stat=0.2369, p=0.0 (expected: p >= 0.05 for normal-like; p < 0.05 -> non-normal -> consider log/power transform)
- **KS test — 'BILL_AMT3' vs normal distribution (p >= 0.05 -> normal-like)** — observed: KS stat=0.24, p=0.0 (expected: p >= 0.05 for normal-like; p < 0.05 -> non-normal -> consider log/power transform)
- **KS test — 'BILL_AMT4' vs normal distribution (p >= 0.05 -> normal-like)** — observed: KS stat=0.2408, p=0.0 (expected: p >= 0.05 for normal-like; p < 0.05 -> non-normal -> consider log/power transform)
- **KS test — 'BILL_AMT5' vs normal distribution (p >= 0.05 -> normal-like)** — observed: KS stat=0.2435, p=0.0 (expected: p >= 0.05 for normal-like; p < 0.05 -> non-normal -> consider log/power transform)
- **KS test — 'BILL_AMT6' vs normal distribution (p >= 0.05 -> normal-like)** — observed: KS stat=0.2458, p=0.0 (expected: p >= 0.05 for normal-like; p < 0.05 -> non-normal -> consider log/power transform)
- **KS test — 'PAY_AMT1' vs normal distribution (p >= 0.05 -> normal-like)** — observed: KS stat=0.3662, p=0.0 (expected: p >= 0.05 for normal-like; p < 0.05 -> non-normal -> consider log/power transform)
- **KS test — 'PAY_AMT2' vs normal distribution (p >= 0.05 -> normal-like)** — observed: KS stat=0.3986, p=0.0 (expected: p >= 0.05 for normal-like; p < 0.05 -> non-normal -> consider log/power transform)
- **KS test — 'PAY_AMT3' vs normal distribution (p >= 0.05 -> normal-like)** — observed: KS stat=0.3833, p=0.0 (expected: p >= 0.05 for normal-like; p < 0.05 -> non-normal -> consider log/power transform)
- **KS test — 'PAY_AMT4' vs normal distribution (p >= 0.05 -> normal-like)** — observed: KS stat=0.379, p=0.0 (expected: p >= 0.05 for normal-like; p < 0.05 -> non-normal -> consider log/power transform)
- **KS test — 'PAY_AMT5' vs normal distribution (p >= 0.05 -> normal-like)** — observed: KS stat=0.3767, p=0.0 (expected: p >= 0.05 for normal-like; p < 0.05 -> non-normal -> consider log/power transform)
- **KS test — 'PAY_AMT6' vs normal distribution (p >= 0.05 -> normal-like)** — observed: KS stat=0.3846, p=0.0 (expected: p >= 0.05 for normal-like; p < 0.05 -> non-normal -> consider log/power transform)
- **KS test — 'avg_bill' vs normal distribution (p >= 0.05 -> normal-like)** — observed: KS stat=0.234, p=0.0 (expected: p >= 0.05 for normal-like; p < 0.05 -> non-normal -> consider log/power transform)
- **KS test — 'avg_payment' vs normal distribution (p >= 0.05 -> normal-like)** — observed: KS stat=0.3014, p=0.0 (expected: p >= 0.05 for normal-like; p < 0.05 -> non-normal -> consider log/power transform)

### Relationships Profile
- **Pearson vs Spearman: 'LIMIT_BAL' ↔ 'avg_bill' (Δ > 0.1 -> non-linear relationship)** — observed: Pearson=0.302, Spearman=0.0919, Δ=0.210 (expected: Δ <= 0.1 (linear); Δ > 0.1 signals non-linear -> use Spearman)
- **Pearson vs Spearman: 'avg_bill' ↔ 'avg_payment' (Δ > 0.1 -> non-linear relationship)** — observed: Pearson=0.3439, Spearman=0.5973, Δ=0.253 (expected: Δ <= 0.1 (linear); Δ > 0.1 signals non-linear -> use Spearman)
- **Pearson vs Spearman: 'total_delinquency' ↔ 'max_delinquency' (Δ > 0.1 -> non-linear relationship)** — observed: Pearson=0.8493, Spearman=0.9883, Δ=0.139 (expected: Δ <= 0.1 (linear); Δ > 0.1 signals non-linear -> use Spearman)
- **No highly collinear feature pairs (|Pearson r| > 0.9) — destabilises models** — observed: 6 pairs: ['BILL_AMT1↔BILL_AMT2 (r=0.951)', 'BILL_AMT2↔BILL_AMT3 (r=0.928)', 'BILL_AMT3↔BILL_AMT4 (r=0.924)', 'BILL_AMT4↔BILL_AMT5 (r=0.940)', 'BILL_AMT4↔BILL_AMT6 (r=0.901)'] (expected: 0 pairs with |r| > 0.9)


---

## Notes for Transformation Step

- avg_macro_CPI, avg_macro_GDP, avg_macro_TAIEX, avg_macro_rate, avg_macro_unemp are CONSTANT across all 30,000 borrowers (same 6-month window for everyone). Their values are accurate as confirmed above, but they carry zero variance for modelling. The transformation step must replace them with interaction features (e.g. CPI only during this borrower's delinquent months).

- PAY_STATUS columns (PAY_0, PAY_2-PAY_6) use the raw UCI encoding (-2 to 8). The transformation step must recode these before feeding them to any model that assumes ordinal monotonicity. -2 (no consumption) and -1 (paid in full) both mean no delinquency but are numerically negative.

- EDUCATION and MARRIAGE contain undocumented codes (0, 5, 6 for EDUCATION; 0 for MARRIAGE). The transformation step must decide whether to remap or group them.

- Distribution profile reveals that BILL_AMT and PAY_AMT columns are likely highly right-skewed (positive skew, leptokurtic). The transformation step should consider log1p or power transforms.

- Isolation Forest detected ~5% multivariate anomalies. Review these borrowers before deciding whether to treat them as legitimate extreme cases or data errors.
