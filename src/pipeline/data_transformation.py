import pandas as pd
import logging
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import PowerTransformer

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


BILL_AMT_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_STATUS_COLS = ["PAY_1", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
TARGET_COL = "default payment next month"
POWER_TRANSFORM_COLS = ["LIMIT_BAL", "AGE"]
PAY_AMT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]

LOG1P_COLS = PAY_AMT_COLS + BILL_AMT_COLS + ["avg_bill", "avg_payment"]


def remap_categorical_codes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remap undocumented / invalid category codes to a documented 'Other' bucket.

    EDUCATION uses codes 0, 5, 6 (not documented in the original UCI paper).
        Documented values: 1=graduate school, 2=university, 3=high school, 4=others.
        Codes 0, 5, 6 → remapped to 4 ('others').

    MARRIAGE uses code 0 (not documented).
        Documented values: 1=married, 2=single, 3=others.
        Code 0 → remapped to 3 ('others').
    """
    df = df.copy()

    if "EDUCATION" in df.columns:
        undoc_edu = df["EDUCATION"].isin([0, 5, 6])
        n = undoc_edu.sum()
        df.loc[undoc_edu, "EDUCATION"] = 4
        log.info("EDUCATION: remapped %d undocumented codes (0/5/6) → 4 (others).", n)

    if "MARRIAGE" in df.columns:
        undoc_mar = df["MARRIAGE"] == 0
        n = undoc_mar.sum()
        df.loc[undoc_mar, "MARRIAGE"] = 3
        log.info("MARRIAGE: remapped %d undocumented codes (0) → 3 (others).", n)

    return df


def recode_pay_status(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recode PAY status columns to a monotonically ordinal scale.

    Problem
    -------
    The raw UCI encoding for PAY_0, PAY_2–PAY_6 mixes two semantically
    distinct concepts in the negative range:
        -2  → no consumption (account inactive that month)
        -1  → paid in full / paid duly
         0  → use of revolving credit (minimum payment made)
         1  → payment delayed 1 month
         2  → payment delayed 2 months
         ...
         8  → payment delayed 8 months

    Both -2 and -1 represent "no delinquency" but are numerically negative.
    Any model that assumes ordinal monotonicity (logistic regression, linear
    SVM, distance-based methods) will misinterpret -2 as *worse* than -1
    or treat the jump from -2 → 1 as a 3-unit step instead of a 1-unit
    delinquency onset, corrupting learned weights/distances.

    Recoding scheme (maps to a clean 0-based ordinal)
    --------------------------------------------------
        Raw -2  →  0   (no consumption — no delinquency)
        Raw -1  →  0   (paid in full   — no delinquency)
        Raw  0  →  1   (revolving credit, min payment)
        Raw  1  →  2   (1 month late)
        Raw  2  →  3   (2 months late)
        ...
        Raw  k  →  k+1 (k months late, k ≥ 1)

    This gives a strictly non-decreasing ordinal where 0 = "on time /
    no delinquency" and higher integers = more severe delay, with no
    negative values to confuse monotonic models.

    No fitting is required — the mapping is a fixed domain rule —
    so this step is safe to apply independently to each split.
    """
    df = df.copy()
    present = [c for c in PAY_STATUS_COLS if c in df.columns]
    if not present:
        log.warning("recode_pay_status: no PAY_STATUS_COLS found — skipped.")
        return df

    for col in present:
        before_neg = int((df[col] < 0).sum())
        # -2 and -1 → 0 (no delinquency); 0 → 1; k ≥ 1 → k+1
        df[col] = df[col].apply(lambda x: 0 if x <= -1 else int(x) + 1)
        after_neg = int((df[col] < 0).sum())
        log.info(
            "recode_pay_status [%s]: %d negative values before → %d after; "
            "new range [%d, %d].",
            col,
            before_neg,
            after_neg,
            int(df[col].min()),
            int(df[col].max()),
        )

    log.info(
        "PAY_STATUS recoding complete: columns recoded = %s. "
        "Mapping: {-2,-1} → 0 (no delinquency), 0 → 1, k → k+1.",
        present,
    )
    return df


def engineer_credit_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add polynomial / interaction features for LIMIT_BAL ↔ avg_bill.

    Spearman correlation analysis (Phase 1) confirmed a non-linear
    relationship (Δ = 0.210) between LIMIT_BAL and avg_bill.
    Linear models cannot capture this without explicit polynomial terms.

    Features added
    --------------
    utilisation_ratio  : avg_bill / (LIMIT_BAL + 1)
        Normalises billing amount by credit limit; the +1 guard
        prevents division by zero for zero-limit edge cases.

    LIMIT_BAL_sq       : LIMIT_BAL ** 2
        Squared term lets a linear model approximate the curvature
        of the LIMIT_BAL ↔ default relationship.

    limit_x_bill       : LIMIT_BAL * avg_bill
        Interaction term capturing the joint effect of credit
        limit and billing behaviour.

    No fitting is required — all operations are fixed arithmetic —
    so this step is safe to apply independently to each split.
    """
    df = df.copy()

    if "avg_bill" in df.columns and "LIMIT_BAL" in df.columns:
        df["utilisation_ratio"] = df["avg_bill"] / (df["LIMIT_BAL"] + 1)
        df["LIMIT_BAL_sq"] = df["LIMIT_BAL"] ** 2
        df["limit_x_bill"] = df["LIMIT_BAL"] * df["avg_bill"]
        log.info(
            "Credit features added: utilisation_ratio, LIMIT_BAL_sq, limit_x_bill."
        )
    else:
        log.warning(
            "engineer_credit_features: 'avg_bill' or 'LIMIT_BAL' missing — skipped."
        )

    return df


def engineer_payment_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add repayment-discipline features for avg_bill ↔ avg_payment.

    Spearman correlation analysis (Phase 1) confirmed a non-linear
    relationship (Δ = 0.253) between avg_bill and avg_payment.
    Tree models handle this natively; linear models benefit from
    the explicit derived signals below.

    Features added
    --------------
    payment_ratio       : avg_payment / (avg_bill + 1)
        Fraction of the average bill that is actually repaid each month.
        The most direct signal of repayment discipline.
        The +1 guard prevents division by zero for zero-bill months.

    avg_unpaid_balance  : avg_bill − avg_payment
        Mean unpaid amount per month; captures accumulating debt burden.
        Negative values (overpayment) are valid and informative.

    is_underpaying      : 1 if avg_payment < avg_bill, else 0
        Binary flag for persistent underpayment, i.e. revolving debt
        is growing on average.

    No fitting is required — all operations are fixed arithmetic —
    so this step is safe to apply independently to each split.
    """
    df = df.copy()

    if "avg_payment" in df.columns and "avg_bill" in df.columns:
        denominator = df["avg_bill"].replace(0, 1)
        df["payment_ratio"] = df["avg_payment"] / denominator
        df["avg_unpaid_balance"] = df["avg_bill"] - df["avg_payment"]
        df["is_underpaying"] = (df["avg_payment"] < df["avg_bill"]).astype(int)
        log.info(
            "Payment features added: payment_ratio, avg_unpaid_balance, is_underpaying."
        )
    else:
        log.warning(
            "engineer_payment_features: 'avg_payment' or 'avg_bill' missing — skipped."
        )

    return df


def engineer_macro_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create meaningful macro interaction features using behavioral signals.

    Strategy:
    - Replace zero-variance macro features with interaction features
    - Use continuous behavior (total_delinquency, payment_ratio, utilisation_ratio)
    - Add one normalized risk feature (capacity-aware)

    Created features:
        cpi_risk_norm        = CPI × total_delinquency / LIMIT_BAL
        captures -> Is this person struggling under high cost of living relative to their credit capacity?

        gdp_x_payment_ratio  = GDP × payment_ratio
        captures -> Is this person paying well because the economy is strong?

        taiex_x_utilisation  = TAIEX × utilisation_ratio
        captures -> Is this person overusing credit in a weak market?

        rate_x_delinquency   = interest_rate × total_delinquency
        Is this person struggling more because borrowing is expensive?

        unemp_x_delinquency  = unemployment × total_delinquency
        Is this person risky because the job market is bad?
    """

    df = df.copy()

    # ===============================
    # Step 1: Check macro columns
    # ===============================
    macro_cols = [
        c
        for c in [
            "avg_macro_CPI",
            "avg_macro_GDP",
            "avg_macro_TAIEX",
            "avg_macro_rate",
            "avg_macro_unemp",
        ]
        if c in df.columns
    ]

    if not macro_cols:
        log.info("engineer_macro_interactions: no macro columns found — skipped.")
        return df

    created = []

    # Step 2: CPI -> normalized  why this one is normalizesd ? becauase avg cpi is 195.1333 so it can explode in magnitude.

    if "avg_macro_CPI" in df.columns and "total_delinquency" in df.columns:
        df["cpi_risk_norm"] = (
            df["avg_macro_CPI"] * df["total_delinquency"] / (df["LIMIT_BAL"] + 1)
        )
        created.append("cpi_risk_norm")

    # Step 3: GDP -> payment behavior

    if "avg_macro_GDP" in df.columns and "payment_ratio" in df.columns:
        df["gdp_x_payment_ratio"] = df["avg_macro_GDP"] * df["payment_ratio"]
        created.append("gdp_x_payment_ratio")

    # Step 4: TAIEX -> utilisation

    if "avg_macro_TAIEX" in df.columns and "utilisation_ratio" in df.columns:
        df["taiex_x_utilisation"] = df["avg_macro_TAIEX"] * df["utilisation_ratio"]
        created.append("taiex_x_utilisation")

    # Step 5: Interest rate -> delinquency

    if "avg_macro_rate" in df.columns and "total_delinquency" in df.columns:
        df["rate_x_delinquency"] = df["avg_macro_rate"] * df["total_delinquency"]
        created.append("rate_x_delinquency")

    # Step 6: Unemployment -> delinquency

    if "avg_macro_unemp" in df.columns and "total_delinquency" in df.columns:
        df["unemp_x_delinquency"] = df["avg_macro_unemp"] * df["total_delinquency"]
        created.append("unemp_x_delinquency")

    # Step 7: Drop raw macro columns

    df = df.drop(columns=macro_cols)

    log.info(
        "Macro interactions: dropped %s; created %s.",
        macro_cols,
        created,
    )

    return df


def standardize_macro_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Standardise the macro interaction columns produced by
    engineer_macro_interactions().

    The scaler is fit ONLY on the training split to prevent leakage
    of val/test statistics into the training distribution.

    If none of the expected columns are present the function is a no-op.
    """
    macro_interaction_cols = [
        c
        for c in [
            "cpi_risk_norm",
            "gdp_x_payment_ratio",
            "taiex_x_utilisation",
            "rate_x_delinquency",
            "unemp_x_delinquency",
        ]
        if c in train_df.columns
    ]

    if not macro_interaction_cols:
        log.info(
            "standardize_macro_features: no macro interaction columns found — skipped."
        )
        return train_df, val_df, test_df

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    scaler = StandardScaler()
    train_df[macro_interaction_cols] = scaler.fit_transform(
        train_df[macro_interaction_cols]
    )
    val_df[macro_interaction_cols] = scaler.transform(val_df[macro_interaction_cols])
    test_df[macro_interaction_cols] = scaler.transform(test_df[macro_interaction_cols])

    log.info(
        "standardize_macro_features: fit on train, applied to all splits: %s.",
        macro_interaction_cols,
    )
    return train_df, val_df, test_df


def handle_bill_amt_collinearity(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    strategy: str = "pca",
    n_components: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Handle the high multicollinearity among BILL_AMT1-6 (|r| > 0.9).

    Root cause: month-over-month balance rollover means consecutive
    bill amounts are nearly identical (Phase 1 §3.9).

    Strategy "pca":
        Compress BILL_AMT1-6 into n_components principal components.
        PCA is fit on training data ONLY, then applied to val and test
        using the same fitted object to prevent data leakage.
        2 components typically capture >95% of variance.

    Strategy "drop":
        Keep only BILL_AMT1 (most recent, most predictive).
        Drop BILL_AMT2-6. Preferred for tree models.
    """
    present = [c for c in BILL_AMT_COLS if c in train_df.columns]
    if not present:
        log.warning("handle_bill_amt_collinearity: no BILL_AMT columns found.")
        return train_df, val_df, test_df

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    if strategy == "drop":
        cols_to_drop = [c for c in present if c != "BILL_AMT1"]
        train_df = train_df.drop(columns=cols_to_drop)
        val_df = val_df.drop(columns=cols_to_drop)
        test_df = test_df.drop(columns=cols_to_drop)
        log.info("BILL_AMT collinearity: dropped %s, kept BILL_AMT1.", cols_to_drop)
        return train_df, val_df, test_df

    if strategy == "pca":
        pca = PCA(n_components=n_components, random_state=42)

        # FIT on train only — never fit on val or test
        train_components = pca.fit_transform(train_df[present])
        val_components = pca.transform(val_df[present])
        test_components = pca.transform(test_df[present])

        explained = pca.explained_variance_ratio_
        log.info(
            "BILL_AMT PCA: %d components explain %.2f%% of variance. Per component: %s",
            n_components,
            sum(explained) * 100,
            [f"PC{i + 1}={v:.3f}" for i, v in enumerate(explained)],
        )

        component_cols = [f"BILL_AMT_PC{i + 1}" for i in range(n_components)]

        train_df = train_df.drop(columns=present)
        val_df = val_df.drop(columns=present)
        test_df = test_df.drop(columns=present)

        train_pca = train_df.copy()
        val_pca = val_df.copy()
        test_pca = test_df.copy()

        for j, col_name in enumerate(component_cols):
            train_pca[col_name] = train_components[:, j]
            val_pca[col_name] = val_components[:, j]
            test_pca[col_name] = test_components[:, j]

        log.info(
            "BILL_AMT collinearity: replaced %d cols with %d PCA components.",
            len(present),
            n_components,
        )
        return train_pca, val_pca, test_pca

    raise ValueError(f"strategy must be 'pca' or 'drop', got '{strategy}'")


def flag_anomalies(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str = TARGET_COL,
    contamination: float = 0.05,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Detect multivariate anomalies with Isolation Forest.

    Phase 1 analysis identified ~5% anomalous records (≈1,500 rows)
    whose feature combinations fall outside the expected data manifold.
    These borrowers should be reviewed before a decision is made to
    retain or exclude them from the training set.

    Policy implemented here (conservative / non-destructive):
        • Fit Isolation Forest on TRAINING features only.
        • Add a binary column `is_anomaly` (1 = anomalous) to all splits.
        • Do NOT automatically drop any rows.
        • The downstream modelling step can choose to:
            - exclude anomalous rows from training,
            - keep them and let the model learn from them, or
            - inspect them separately.

    contamination=0.05 matches the Phase 1 estimate of ~5% anomalies.

    The fitter uses only feature columns (target_col excluded) so that
    the label cannot influence anomaly scoring (no target leakage).
    """
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    feature_cols = [c for c in train_df.columns if c != target_col]

    iso = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )

    # FIT on train features only
    iso.fit(train_df[feature_cols])

    for df_part, label in [(train_df, "train"), (val_df, "val"), (test_df, "test")]:
        feat_cols_part = [c for c in feature_cols if c in df_part.columns]
        # Isolation Forest returns -1 (anomaly) / +1 (normal); recode to 0/1
        raw_pred = iso.predict(df_part[feat_cols_part])
        df_part["is_anomaly"] = (raw_pred == -1).astype(int)
        n_anomalies = df_part["is_anomaly"].sum()
        log.info(
            "Anomaly detection [%s]: %d anomalous rows flagged (%.2f%%).",
            label,
            n_anomalies,
            100 * n_anomalies / len(df_part),
        )

    return train_df, val_df, test_df


def handle_data_imbalance(
    train_df: pd.DataFrame,
    target_col: str = TARGET_COL,
    random_state: int = 42,
    k_neighbors: int = 5,
) -> pd.DataFrame:
    """
    Apply SMOTE to balance the training set.

    SMOTE must be the LAST transformation step because:
      1. It generates synthetic samples by interpolating between real
         training neighbours. Applying it before feature engineering
         would force synthetic rows through transformers fitted on a
         different (unaugmented) distribution.
      2. It must NEVER touch val or test — their class distribution
         must mirror the real population for honest evaluation.
      3. Applying it before PCA / scalers would leak the synthetic
         distribution into the fitted transform objects.

    The 'is_anomaly' flag column (if present) is treated as a regular
    feature here — SMOTE does not receive the target column.
    """
    log.info("Applying SMOTE on training dataframe...")

    if target_col not in train_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]

    smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    X_resampled = pd.DataFrame(X_resampled, columns=X_train.columns)
    y_name = y_train.name if hasattr(y_train, "name") and y_train.name else target_col
    y_resampled = pd.Series(y_resampled, name=y_name)

    resampled_df = X_resampled.copy()
    resampled_df[y_name] = y_resampled

    log.info("Class distribution before SMOTE: %s", y_train.value_counts().to_dict())
    log.info(
        "Class distribution after  SMOTE: %s", y_resampled.value_counts().to_dict()
    )

    return resampled_df


def handle_distribution(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Apply Yeo-Johnson PowerTransformer to LIMIT_BAL and AGE.

    The transformer is fit ONLY on the training split to learn the lambda
    parameters, then the same fitted transformer is used to transform
    val and test — preventing any leakage of test statistics.

    Transformation (fixed function, no fitting):
        Apply log1p to all PAY_AMT, BILL_AMT, avg_bill, avg_payment columns.
        log1p is safe for zero-heavy financial data (log(1+0)=0).
    """
    present = [c for c in POWER_TRANSFORM_COLS if c in train_df.columns]
    if not present:
        log.info("Distribution: no power-transform columns found — skipping.")
        return train_df, val_df, test_df

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    transformer = PowerTransformer(method="yeo-johnson", standardize=True)

    # FIT on train only
    transformer.fit(train_df[present])

    # APPLY to all splits
    train_df[present] = transformer.transform(train_df[present])
    val_df[present] = transformer.transform(val_df[present])
    test_df[present] = transformer.transform(test_df[present])

    log.info(
        "Distribution: Yeo-Johnson PowerTransformer fit on train, applied to all splits: %s.",
        present,
    )

    # --- log1p transform (fixed function — no fitting needed) ---
    log.info("Outliers applying log1p transform to amount columns ...")
    for df_part, label in [(train_df, "train"), (val_df, "val"), (test_df, "test")]:
        for col in LOG1P_COLS:
            if col not in df_part.columns:
                continue
            min_val = df_part[col].min()
            if min_val < 0:
                # Shift so minimum = 0 before log;
                df_part[col] = df_part[col] - min_val
                log.info(
                    "  %s [%s]: shifted by %.2f before log1p.", col, label, min_val
                )
            df_part[col] = np.log1p(df_part[col])

    log.info("log1p applied to amount columns in all splits.")
    return train_df, val_df, test_df


def run_transformation(
    train_input: str,
    val_input: str,
    test_input: str,
    train_output: str,
    val_output: str,
    test_output: str,
) -> None:
    """
    transformation pipeline.

    Step order
    -------------------------
    1.  Remap undocumented categorical codes
    2.  Engineer credit-utilisation features
    3.  Engineer payment-behaviour features
    4.  Engineer macro interaction features
    5.  Standardise macro interaction features        (fit on train only)
    6. Handle skewed distributions with PowerTransformer  and log1p   (fit on train only)
    7.  Handle BILL_AMT collinearity (PCA or drop)   (fit on train only)
    8.  Flag multivariate anomalies (Isolation Forest)(fit on train only)
    9.  SMOTE oversampling                            (train only, LAST step)

    Val and test sets are NEVER used to fit any transformer.
    SMOTE is applied last so synthetic samples do not contaminate
    any fitted transformer's internal state. a.ka learn from fake data
    """

    log.info("Loading cleaned datasets...")
    train_df = pd.read_csv(train_input)
    val_df = pd.read_csv(val_input)
    test_df = pd.read_csv(test_input)

    log.info("--- Step 1: Remap undocumented categorical codes ---")
    train_df = remap_categorical_codes(train_df)
    val_df = remap_categorical_codes(val_df)
    test_df = remap_categorical_codes(test_df)

    log.info("--- Step 2: Recode PAY_STATUS columns to ordinal scale ---")
    train_df = recode_pay_status(train_df)
    val_df = recode_pay_status(val_df)
    test_df = recode_pay_status(test_df)

    log.info("--- Step 3: Engineer credit-utilisation features ---")
    train_df = engineer_credit_features(train_df)
    val_df = engineer_credit_features(val_df)
    test_df = engineer_credit_features(test_df)

    log.info("--- Step 4: Engineer payment-behaviour features ---")
    train_df = engineer_payment_features(train_df)
    val_df = engineer_payment_features(val_df)
    test_df = engineer_payment_features(test_df)
    mask = train_df["payment_ratio"].isna()
    log.info(
        "Problematic rows:\n%s",
        train_df[mask][["avg_payment", "avg_bill", "payment_ratio"]].to_string(),
    )
    log.info("avg_payment dtype: %s", train_df["avg_payment"].dtype)
    log.info("avg_bill dtype: %s", train_df["avg_bill"].dtype)
    log.info("--- Step 5: Engineer macro interaction features ---")
    train_df = engineer_macro_interactions(train_df)
    val_df = engineer_macro_interactions(val_df)
    test_df = engineer_macro_interactions(test_df)

    log.info(
        "--- Step 6: Handle skewed distributions with PowerTransformer and log1p ---"
    )
    train_df, val_df, test_df = handle_distribution(train_df, val_df, test_df)

    log.info("--- Step 7: Standardise macro interaction features ---")
    train_df, val_df, test_df = standardize_macro_features(train_df, val_df, test_df)

    log.info("--- Step 8: Handle BILL_AMT collinearity (%s) ---", "pca")
    train_df, val_df, test_df = handle_bill_amt_collinearity(
        train_df,
        val_df,
        test_df,
        strategy="pca",
        n_components=2,
    )

    log.info("--- Step 9: Flag multivariate anomalies (Isolation Forest) ---")
    train_df, val_df, test_df = flag_anomalies(
        train_df,
        val_df,
        test_df,
        contamination=0.05,
    )
    for split, df_part in [("train", train_df), ("val", val_df), ("test", test_df)]:
        nan_pay = df_part["avg_payment"].isna().sum()
        nan_bill = df_part["avg_bill"].isna().sum()
        nan_ratio = (
            df_part["payment_ratio"].isna().sum()
            if "payment_ratio" in df_part.columns
            else "col missing"
        )
        log.info(
            "[%s] avg_payment NaN=%s | avg_bill NaN=%s | payment_ratio NaN=%s",
            split,
            nan_pay,
            nan_bill,
            nan_ratio,
        )
    Path(val_output).parent.mkdir(parents=True, exist_ok=True)
    Path(test_output).parent.mkdir(parents=True, exist_ok=True)
    val_df.to_csv(val_output, index=False)
    test_df.to_csv(test_output, index=False)
    log.info("Saved val  -> %s", val_output)
    log.info("Saved test -> %s", test_output)

    log.info("--- Step 9: SMOTE (training split only, last step) ---")
    train_df = handle_data_imbalance(train_df)

    Path(train_output).parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(train_output, index=False)
    log.info("Saved train -> %s", train_output)

def check_required_files(paths):
    missing = [p for p in paths if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files: {missing}")
    
if __name__ == "__main__":
    try:
        train_path = "data/clean/train_cleaned.csv"
        val_path = "data/clean/val_cleaned.csv"
        test_path = "data/clean/test_cleaned.csv"

        check_required_files([train_path, val_path, test_path])

        run_transformation(
            train_input=train_path,
            val_input=val_path,
            test_input=test_path,
            train_output="data/transformed/train_transformed.csv",
            val_output="data/transformed/val_transformed.csv",
            test_output="data/transformed/test_transformed.csv",
        )

        log.info("Transformation completed successfully.")

    except Exception as e:
        log.exception(f"Transformation script failed: {e}")
        raise   

