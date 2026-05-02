from loguru import logger
import pandas as pd  # Data manipulation and analysis
import numpy as np  # Numerical computing
import matplotlib.pyplot as plt  # Core plotting (the matplotlib.pyplot module)
from scipy import stats
import seaborn as sns  # Statistical visualizations (built on matplotlib)
from pathlib import Path
from scipy.stats import pearsonr, linregress
from src.pipeline.config import (
    test_transformed_path_without_smote,
    train_transformed_path_without_smote,
    val_transformed_path_without_smote,
)

plt.style.use("dark_background")

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s | %(levelname)s | %(message)s",
# )
# log = logging.getLogger(__name__)


class EDA:
    def __init__(
        self,
        train_input_cleaned,
        val_input_cleaned,
        test_input_cleaned,
        train_input_transformed,
        val_input_transformed,
        test_input_transformed,
        train_input_transformed_without_smote=train_transformed_path_without_smote,
        val_input_transformed_without_smote=val_transformed_path_without_smote,
        test_input_transformed_without_smote=test_transformed_path_without_smote,
    ):
        logger.info("EDA Module has started")
        self.full_df_transformed = None
        self.full_df_cleaned = None
        self.full_df_transformed_without_smote = None

        self.train_input_cleaned = train_input_cleaned
        self.val_input_cleaned = val_input_cleaned
        self.test_input_cleaned = test_input_cleaned

        self.train_input_transformed = train_input_transformed
        self.val_input_transformed = val_input_transformed
        self.test_input_transformed = test_input_transformed

        self.train_input_transformed_without_smote = (
            train_input_transformed_without_smote
        )
        self.val_input_transformed_without_smote = val_input_transformed_without_smote
        self.test_input_transformed_without_smote = test_input_transformed_without_smote
        self.discrete_features = [
            "default payment next month",
            "SEX",
            "EDUCATION",
            "MARRIAGE",
            "is_anomaly",
            "is_underpaying",
        ]
        self.mapping = {
            "default payment next month": ["No Default", "Default"],
            "SEX": ["Male", "Female"],
            "MARRIAGE": ["married", "single", "other"],
            "EDUCATION": ["graduate", "university", "high school", "other"],
            "is_anomaly": ["Normal", "Anomaly"],
            "is_underpaying": ["not_underpaying", "is_underpaying"],
        }
        self.continuous_features = [
            "LIMIT_BAL",
            "AGE",
            "PAY_1",
            "PAY_2",
            "PAY_3",
            "PAY_4",
            "PAY_5",
            "PAY_6",
            "PAY_AMT1",
            "PAY_AMT2",
            "PAY_AMT3",
            "PAY_AMT4",
            "PAY_AMT5",
            "PAY_AMT6",
            "avg_bill",
            "avg_payment",
            "max_delinquency",
            "total_delinquency",
            "utilisation_ratio",
            "LIMIT_BAL_sq",
            "limit_x_bill",
            "payment_ratio",
            "avg_unpaid_balance",
            "is_underpaying",
            "cpi_risk_norm",
            "gdp_x_payment_ratio",
            "taiex_x_utilisation",
            "rate_x_delinquency",
            "unemp_x_delinquency",
            "BILL_AMT_PC1",
            "BILL_AMT_PC2",
        ]

    def load_data_transformed(self):
        logger.info("This loader for transformed data eda")
        logger.info("train data is loaded")
        train_dataset = pd.read_csv(self.train_input_transformed)
        logger.info("val data is loaded")
        val_dataset = pd.read_csv(self.val_input_transformed)
        logger.info("test data is loaded")
        test_dataset = pd.read_csv(self.test_input_transformed)
        logger.info(f"train shape is: {train_dataset.shape}")
        logger.info(f"val shape is: {val_dataset.shape}")
        logger.info(f"test shape is: {test_dataset.shape}")
        full_df = pd.concat(
            [train_dataset, val_dataset, test_dataset], ignore_index=True
        )
        # ignore_index=True --> to combine index from 0 to N  instead of duplicates
        self.full_df_transformed = full_df
        logger.info("full data has being created and ready for EDA")
        logger.success(f"full data shape is: {full_df.shape}")
        return train_dataset, val_dataset, test_dataset, full_df

    def load_data_cleaned(self):
        logger.info("This loader for cleaned data eda")
        logger.info("train data is loaded")
        train_dataset = pd.read_csv(self.train_input_cleaned)
        logger.info("val data is loaded")
        val_dataset = pd.read_csv(self.val_input_cleaned)
        logger.info("test data is loaded")
        test_dataset = pd.read_csv(self.test_input_cleaned)

        print(f"train shape is: {train_dataset.shape}")
        print(f"val shape is: {val_dataset.shape}")
        print(f"test shape is: {test_dataset.shape}")
        full_df = pd.concat(
            [train_dataset, val_dataset, test_dataset], ignore_index=True
        )
        # ignore_index=True --> to combine index from 0 to N  instead of duplicates
        self.full_df_cleaned = full_df
        logger.success(
            f"full data has being created and ready for EDA with shape: {full_df.shape}"
        )
        return train_dataset, val_dataset, test_dataset, full_df

    def load_data_transformed_without_smote(self):
        logger.info("This loader for transformed without smote data eda")
        logger.info("train data is loaded")
        train_dataset = pd.read_csv(self.train_input_transformed_without_smote)
        logger.info("val data is loaded")
        val_dataset = pd.read_csv(self.val_input_transformed_without_smote)
        logger.info("test data is loaded")
        test_dataset = pd.read_csv(self.test_input_transformed_without_smote)
        logger.info(f"train shape is: {train_dataset.shape}")
        logger.info(f"val shape is: {val_dataset.shape}")
        logger.info(f"test shape is: {test_dataset.shape}")
        full_df = pd.concat(
            [train_dataset, val_dataset, test_dataset], ignore_index=True
        )
        # ignore_index=True --> to combine index from 0 to N  instead of duplicates
        self.full_df_transformed_without_smote = full_df
        logger.info("full data has being created and ready for EDA")
        logger.success(f"full data shape is: {full_df.shape}")
        return train_dataset, val_dataset, test_dataset, full_df

    def apply_univariate(self, column_name, data_type=0):
        # ─── 1. UNIVARIATE: Distribution of Limit Bal ──────────────────────────
        # WHY: Price is right-skewed (few very expensive orders).
        # Histogram + KDE together show both frequency and shape.
        if data_type == 1:
            full_df = self.full_df_cleaned.copy()
        elif data_type == 0:
            full_df = self.full_df_transformed.copy()
        else:
            full_df = self.full_df_transformed_without_smote.copy()
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        # Creates a figure with 1 row and 3 columns of subplots
        # axes[0] = histogram, axes[1] = KDE, axes[2] = box plot

        # ─── Plot 1a: Histogram ────────────────────────────────────────────────────
        axes[0].hist(
            full_df[column_name].dropna(),  # Remove NaN values before plotting
            bins=50,  # Number of bins (try 20–100 for continuous data)
            color="#CF27D4",  # Blue fill color (hex code)
            alpha=0.8,  # 80% opacity — slightly transparent
            edgecolor="white",  # White lines between bars for visual separation
        )
        axes[0].set_title(
            f"{column_name} Distribution (Histogram)", fontsize=12, fontweight="bold"
        )
        axes[0].set_xlabel(column_name)  # BRL = Brazilian Real
        axes[0].set_ylabel("Frequency")  # Y-axis = count of orders in each bin

        # ─── Plot 1b: KDE ─────────────────────────────────────────────────────────
        sns.kdeplot(
            full_df[column_name].dropna(),  # The data series
            ax=axes[1],  # Which subplot to draw on
            color="#577BF1",  # Green color
            fill=True,  # Shade the area under the curve
            alpha=0.84,  # Transparency for the filled area
        )
        axes[1].set_title(f"{column_name} KDE", fontsize=12, fontweight="bold")
        axes[1].set_xlabel(column_name)
        # No ylabel needed — KDE y-axis is "density" (probability per unit)

        # ─── Plot 1c: Box Plot ─────────────────────────────────────────────────────
        axes[2].boxplot(
            full_df[column_name].dropna(),
            vert=False,  # Horizontal layout (easier to read long tails)
            patch_artist=True,  # Required to fill the box with color
            boxprops=dict(
                facecolor="#EBF4FB",
                color="#1B6CA8",  # Light blue box fill  # Box border color
            ),
            medianprops=dict(
                color="#D63031",  # Red median line (stands out clearly)
                linewidth=2,  # Thick median line
            ),
        )
        axes[2].set_title(f"{column_name} Box Plot", fontsize=12, fontweight="bold")
        axes[2].set_xlabel(column_name)

        plt.tight_layout()  # Adjusts spacing between subplots to prevent overlap
        # plt.savefig('olist_univariate_price.png', dpi=150, bbox_inches='tight')
        plt.show()

        # ─── Print summary statistics ──────────────────────────────────────────────
        mean_price = full_df[column_name].mean()
        median_price = full_df[column_name].median()
        print(f"Mean {column_name}: {mean_price:.2f}")
        print(f"Median {column_name}: {median_price:.2f}")
        print(
            f"Skew indicator: mean {'>' if mean_price > median_price else '<'} median → {'RIGHT' if mean_price > median_price else 'LEFT'}-skewed"
        )
        print()
        print(full_df[column_name].describe().round(2))

    def apply_pie_chart(self, feature, mapping, data_type=0):
        if data_type == 1:
            full_df = self.full_df_cleaned.copy()
        elif data_type == 0:
            full_df = self.full_df_transformed.copy()
        else:
            full_df = self.full_df_transformed_without_smote.copy()
        palette = sns.color_palette("Set2", 8)
        plt.figure(figsize=(4, 4))
        counts_series = full_df[feature].map(mapping)
        counts = counts_series.value_counts()
        num_categories = len(counts)
        indices = np.random.choice(len(palette), size=num_categories, replace=False)
        colors = [palette[i] for i in indices]
        plt.pie(
            counts,
            labels=counts.index,
            autopct="%1.1f%%",
            startangle=140,
            colors=colors,
            wedgeprops=dict(edgecolor="black", linewidth=1),
            textprops={"fontsize": 12},
        )

        plt.title(f"Distribution of {feature}", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.show()

    #######################################################################
    def continues_versus_continuous_eda(self):

        n = len(self.continuous_features)

        for i, row_feat in enumerate(self.continuous_features):
            fig, axes = plt.subplots(1, n, figsize=(n * 5, 5))

            if n == 1:
                axes = [axes]

            fig.suptitle(
                f"Analysis: {row_feat} vs Others",
                fontsize=16,
                fontweight="bold",
                y=1.05,
            )

            for j, col_feat in enumerate(self.continuous_features):
                ax = axes[j]

                if row_feat == col_feat:
                    sns.histplot(
                        self.full_df_transformed[
                            row_feat
                        ].dropna(),  # to help me see the feature distribution
                        kde=True,
                        ax=ax,
                        color="#3834EC",
                        alpha=0.6,
                    )
                    ax.set_title(f"Distribution of {row_feat}", color="#3FEB0B")

                else:
                    mask = (
                        ~self.full_df_transformed[row_feat].isna()
                        & ~self.full_df_transformed[col_feat].isna()
                    )
                    x_data = self.full_df_transformed[col_feat][mask]
                    y_data = self.full_df_transformed[row_feat][mask]

                    if len(x_data) > 0:
                        ax.scatter(x_data, y_data, alpha=0.3, color="#E02EBA", s=15)

                        m, b, r, p, _ = stats.linregress(x_data, y_data)
                        x_line = np.linspace(x_data.min(), x_data.max(), 100)
                        ax.plot(
                            x_line,
                            m * x_line + b,
                            color="#3FEB0B",
                            lw=2,
                            label=f"r={r:.2f}",
                        )
                        ax.legend(loc="upper right")

                ax.set_xlabel(col_feat)
                ax.set_ylabel(row_feat)
                ax.grid(True, linestyle="--", alpha=0.3)

            plt.tight_layout()
            plt.show()

    def continuous_versus_discrete_eda(self, feature):
        n = len(self.discrete_features)
        fig, axes = plt.subplots(n, 3, figsize=(15, 6 * n))

        if n == 1:
            axes = np.expand_dims(axes, axis=0)

        colors = ["#BC12CC", "#3FEB0B", "#3B27F0", "#74B9FF", "#00B894"]

        for i, disc_feat in enumerate(self.discrete_features):
            unique_vals = sorted(self.full_df_transformed[disc_feat].unique())
            data_groups = [
                self.full_df_transformed[self.full_df_transformed[disc_feat] == val][
                    feature
                ].dropna()
                for val in unique_vals
            ]

            ax_box = axes[i, 0]
            bp = ax_box.boxplot(
                data_groups, patch_artist=True, labels=self.mapping[disc_feat]
            )

            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            ax_box.set_title(
                f"Distribution of {feature} by {disc_feat}",
                fontsize=12,
                fontweight="bold",
            )
            ax_box.set_ylabel(feature)

            ax_scatter = axes[i, 1]
            for val, color, label in zip(unique_vals, colors, self.mapping[disc_feat]):
                subset = self.full_df_transformed[
                    self.full_df_transformed[disc_feat] == val
                ]
                ax_scatter.scatter(
                    subset.index,
                    subset[feature],
                    c=color,
                    label=label,
                    alpha=0.3,
                    edgecolors=None,
                    s=30,
                )

            ax_scatter.set_title(
                f"{feature} Scatter colored by {disc_feat}",
                fontsize=12,
                fontweight="bold",
            )
            ax_scatter.set_ylabel(feature)
            ax_scatter.legend()
            ##########################################################
            ax_kde = axes[i, 2]

            for val, color, label in zip(unique_vals, colors, self.mapping[disc_feat]):
                sns.kdeplot(
                    data=self.full_df_transformed[
                        self.full_df_transformed[disc_feat] == val
                    ],
                    x=feature,
                    ax=ax_kde,
                    fill=True,
                    alpha=0.4,
                    color=color,
                    label=label,
                )
            ax_kde.set_title(f"{feature} KDE Overlap", fontweight="bold")
            ax_kde.legend()
            valid_df = self.full_df_transformed[[disc_feat, feature]].dropna()
            r, _ = pearsonr(valid_df[disc_feat], valid_df[feature])
            m, b, _, _, _ = linregress(valid_df[disc_feat], valid_df[feature])

            print("\n" + "-" * 40)
            print(f"{feature} vs {disc_feat}")

            print("Regression:")
            print(f"  {feature} = {m:.3f} × {disc_feat} + {b:.2f}")

            print("Correlation:")
            print(
                f"  r = {r:.3f} → "
                f"{'Weak' if abs(r) < 0.4 else 'Moderate' if abs(r) < 0.7 else 'Strong'} "
                f"{'positive' if r > 0 else 'negative' if r < 0 else 'no'}"
            )

            print("-" * 40 + "\n")

        plt.tight_layout()
        # plt.savefig('olist_bivariate_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()

    def discrete_versus_target_stacked(self, target_column):
        features_to_plot = [f for f in self.discrete_features if f != target_column]

        n = len(features_to_plot)
        fig, axes = plt.subplots(n, 1, figsize=(10, 5 * n))

        if n == 1:
            axes = [axes]  # to become a list

        for i, col in enumerate(features_to_plot):
            ct = pd.crosstab(
                self.full_df_transformed[col],
                self.full_df_transformed[target_column],
                normalize="index",
            )  # to be normalization ratio

            if col in self.mapping:  # to use the mapping as labels instead of numbers
                ct.index = self.mapping[col]
            if target_column in self.mapping:
                ct.columns = self.mapping[target_column]

            ct.plot(
                kind="bar",
                stacked=True,
                ax=axes[i],
                color=["#577BF1", "#CF27D4", "#3FEB0B"],
                alpha=0.85,
                edgecolor="white",
            )

            axes[i].set_title(
                f"Target Distribution by {col}", fontsize=14, fontweight="bold"
            )
            axes[i].set_ylabel("Proportion")
            axes[i].set_xlabel(col)
            axes[i].legend(
                title=target_column, bbox_to_anchor=(1.05, 1), loc="upper left"
            )
            axes[i].tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.show()

    def plot_large_correlation_matrix(self):
        plt.figure(figsize=(20, 18))
        features = self.full_df_transformed.columns
        corr = self.full_df_transformed[features].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))

        sns.heatmap(
            corr,
            annot=False,
            mask=mask,
            cmap="RdBu",
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=0.1,
            cbar_kws={"shrink": 0.8},
        )

        plt.title(f"Correlation Matrix for {len(features)} Features", fontsize=20)
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.show()


def run_eda(
    train_input_cleaned,
    val_input_cleaned,
    test_input_cleaned,
    train_input_transformed,
    val_input_transformed,
    test_input_transformed,
):
    eda = EDA(
        train_input_cleaned,
        val_input_cleaned,
        test_input_cleaned,
        train_input_transformed,
        val_input_transformed,
        test_input_transformed,
    )
    eda.load_data_transformed()
    eda.load_data_cleaned()
    eda.load_data_transformed_without_smote()


if __name__ == "__main__":
    run_eda(
        train_input_cleaned="data/clean/train_cleaned.csv",
        val_input_cleaned="data/clean/val_cleaned.csv",
        test_input_cleaned="data/clean/test_cleaned.csv",
        train_input_transformed="data/transformed/train_transformed.csv",
        val_input_transformed="data/transformed/val_transformed.csv",
        test_input_transformed="data/transformed/test_transformed.csv",
    )

if __name__ != "__main__":
    try:
        if (
            Path("data/clean/test_cleaned.csv").exists()
            and Path("data/clean/train_cleaned.csv").exists()
            and Path("data/clean/val_cleaned.csv").exists()
        ):
            run_eda(
                train_input_cleaned="data/clean/train_cleaned.csv",
                val_input_cleaned="data/clean/val_cleaned.csv",
                test_input_cleaned="data/clean/test_cleaned.csv",
                train_input_transformed="data/transformed/train_transformed.csv",
                val_input_transformed="data/transformed/val_transformed.csv",
                test_input_transformed="data/transformed/test_transformed.csv",
            )
    except Exception as e:
        logger.warning(f"Auto-eda failed during import: {e}")

        logger.warning(f"Auto-eda failed during import: {e}")