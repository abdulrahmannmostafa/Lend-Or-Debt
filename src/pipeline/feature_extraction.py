import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif, VarianceThreshold
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from loguru import logger


class FeatureExtractor:
    def __init__(self):
        self.min_max_scaler = MinMaxScaler()
        self.robust_scaler = RobustScaler()

    def smart_scaler(self, X, flag=True):
        logger.debug("smart scaling has started")
        X_copy = X.copy()

        if "SEX" in X_copy.columns and X_copy["SEX"].max() > 1:
            X_copy["SEX"] = X_copy["SEX"].map({1: 0, 2: 1})

        if "MARRIAGE" in X_copy.columns:
            X_copy = pd.get_dummies(
                X_copy, columns=["MARRIAGE"], drop_first=True, dtype=int
            )

        robust_features = [
            "LIMIT_BAL_sq",
            "PAY_AMT1",
            "PAY_AMT2",
            "PAY_AMT3",
            "PAY_AMT4",
            "PAY_AMT5",
            "PAY_AMT6",
            "avg_bill",
            "avg_payment",
            "avg_unpaid_balance",
            "limit_x_bill",
            "utilisation_ratio",
            "payment_ratio",
            "BILL_AMT_PC1",
            "BILL_AMT_PC2",
        ]

        already_scaled = [
            "LIMIT_BAL",
            "AGE",
            "cpi_risk_norm",
            "gdp_x_payment_ratio",
            "taiex_x_utilisation",
            "rate_x_delinquency",
            "unemp_x_delinquency",
        ]

        minmax_features = [
            "EDUCATION",
            "PAY_1",
            "PAY_2",
            "PAY_3",
            "PAY_4",
            "PAY_5",
            "PAY_6",
            "total_delinquency",
            "max_delinquency",
        ]

        seen_features = set(robust_features + already_scaled + minmax_features)
        binary_features = [col for col in X_copy.columns if col not in seen_features]

        if flag:
            self.robust_scaler = RobustScaler()
            self.minmax_scaler = MinMaxScaler()
            X_robust = self.robust_scaler.fit_transform(X_copy[robust_features])
            X_minmax = self.minmax_scaler.fit_transform(X_copy[minmax_features])
        else:
            X_robust = self.robust_scaler.transform(X_copy[robust_features])
            X_minmax = self.minmax_scaler.transform(X_copy[minmax_features])

        df_robust = pd.DataFrame(X_robust, columns=robust_features, index=X_copy.index)
        df_minmax = pd.DataFrame(X_minmax, columns=minmax_features, index=X_copy.index)

        X_final = pd.concat(
            [df_robust, df_minmax, X_copy[already_scaled], X_copy[binary_features]],
            axis=1,
        )
        logger.success(f"Scaling Complete: Output Columns: {len(X_final.columns)}")
        return X_final

    def remove_low_variance_features(self, X, threshold=0.001):
        selector = VarianceThreshold(threshold=threshold)
        X_new = selector.fit_transform(X)
        selected_cols = X.columns[selector.get_support()]
        return pd.DataFrame(X_new, columns=selected_cols, index=X.index)

    def pearson_correlation(self, X_train, y_train, k=20):
        logger.debug("starting Pearson feature selection method")
        scores = X_train.apply(lambda col: col.corr(y_train, method="pearson")).abs()
        feature_scores = pd.DataFrame(
            {"Feature": X_train.columns, "Score": scores}
        ).sort_values(by="Score", ascending=False)

        selected_features = feature_scores.head(k)["Feature"].tolist()

        logger.debug(f"--- PEARSON TOP {k} ---")
        print(feature_scores.head(k))

        return selected_features, feature_scores

    def spearman_correlation(self, X_train, y_train, k=20):
        logger.debug("starting Spearman feature selection method")
        scores = X_train.apply(lambda col: col.corr(y_train, method="spearman")).abs()
        feature_scores = pd.DataFrame(
            {"Feature": X_train.columns, "Score": scores}
        ).sort_values(by="Score", ascending=False)

        selected_features = feature_scores.head(k)["Feature"].tolist()

        logger.debug(f"--- SPEARMAN TOP {k} ---")
        print(feature_scores.head(k))

        return selected_features, feature_scores

    def select_k_best_mutual_info(self, X_train, y_train, k=20):
        logger.debug("starting mutual_info feature selection method")
        discrete_features = [
            col
            for col in X_train.columns
            if col
            in [
                "SEX",
                "EDUCATION",
                "total_delinquency",
                "max_delinquency",
                "is_anomaly",
            ]
            or col.startswith("MARRIAGE_")
            or (col.startswith("PAY_") and len(col) == 5)
        ]

        discrete_mask = [col in discrete_features for col in X_train.columns]
        mi_scores = mutual_info_classif(
            X_train, y_train, discrete_features=discrete_mask, random_state=42
        )

        feature_scores = pd.DataFrame(
            {"Feature": X_train.columns, "Score": mi_scores}
        ).sort_values(by="Score", ascending=False)

        selected_features = feature_scores.head(k)["Feature"].tolist()

        logger.debug(f"--- MUTUAL INFO TOP {k} ---")
        print(feature_scores.head(k))

        return selected_features, feature_scores

    def test_feature_selection_methods(self, X_train, y_train, k=15):
        logger.debug("test_feature_selection_methods")
        results = {}

        cols_mi, scores_mi = self.select_k_best_mutual_info(X_train, y_train, k=k)
        results["mutual_info"] = {"cols": cols_mi, "scores": scores_mi}

        cols_pe, scores_pe = self.pearson_correlation(X_train, y_train, k=k)
        results["pearson"] = {"cols": cols_pe, "scores": scores_pe}

        cols_sp, scores_sp = self.spearman_correlation(X_train, y_train, k=k)
        results["spearman"] = {"cols": cols_sp, "scores": scores_sp}

        return results

    def apply_pca(self, X_train, X_val, X_test, n_components=0.95):
        logger.debug("starting pca method")
        pca = PCA(n_components=n_components, random_state=42)
        X_train_pca = pca.fit_transform(X_train)
        X_val_pca = pca.transform(X_val)
        X_test_pca = pca.transform(X_test)
        return X_train_pca, X_val_pca, X_test_pca
