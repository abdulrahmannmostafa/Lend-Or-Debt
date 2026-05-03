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
from matplotlib import gridspec

plt.style.use("dark_background")

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s | %(levelname)s | %(message)s",
# )
# log = logging.getLogger(__name__)

ACCENT_BLUE = "#130A69"  # primary accent — orders, neutral info
ACCENT_GREEN = "#2BA08B"  # positive / high ratings
ACCENT_ORANGE = "#91734E"  # order value / weekends
ACCENT_RED = "#690A42"  # negative / low ratings / delivery time
ACCENT_Pink = "#CB7EE2"  # negative / low ratings / delivery time
CARD_BG = "#D0D0E2"  # dark card
CANVAS_BG = "#22222C"  # darker background
BORDER_CLR = "#D8DCE8"  # card border colour
TEXT_DARK = "#C2C8E9"  # headings
TEXT_MID = "#FFFFFF"  # labels
TEXT_MUTED = "#4A5068"  # sub-labels, axis ticks
GRID_CLR = "#EAECF4"  # chart grid lines


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
            "default payment next month": {0: "No Default", 1: "Default"},
            "SEX": {1: "Male", 2: "Female"},
            "MARRIAGE": {1: "married", 2: "single", 3: "other"},
            "EDUCATION": {1: "graduate", 2: "university", 3: "high school", 4: "other"},
            "is_anomaly": {0: "Normal", 1: "Anomaly"},
            "is_underpaying": {0: "not_underpaying", 1: "is_underpaying"},
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

    def continuous_vs_continuous_eda(self, feature_1, feature_2):
        mask = (
            ~self.full_df_transformed[feature_1].isna()
            & ~self.full_df_transformed[feature_2].isna()
        )
        x_data = self.full_df_transformed[feature_1][mask]
        y_data = self.full_df_transformed[feature_2][mask]

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(x_data, y_data, alpha=0.3, color="#E02EBA", s=15)

        if len(x_data) > 0:
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

        ax.set_title(
            f"Analysis: {feature_1} vs {feature_2}", fontsize=16, fontweight="bold"
        )
        ax.set_xlabel(feature_1)
        ax.set_ylabel(feature_2)
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

    # ─────────────────────────────────────────────────────────────────────────
    # HELPER: apply common axis styling shared by all chart panels
    # ─────────────────────────────────────────────────────────────────────────
    def style_axis(self, ax, title, ylabel="", xlabel="", xgrid=False):
        ax.set_facecolor(CARD_BG)
        ax.set_title(title, fontsize=17, fontweight="bold", color=TEXT_DARK, pad=20)
        ax.set_ylabel(ylabel, color=TEXT_MID, fontsize=13)
        ax.set_xlabel(xlabel, color=TEXT_MID, fontsize=13)
        ax.tick_params(colors=TEXT_MID, labelsize=12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for sp in ["left", "bottom"]:
            ax.spines[sp].set_color(BORDER_CLR)
        ax.yaxis.grid(True, color=GRID_CLR, linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)  # ensure grid lines render below bars/lines, not on top

    # ─────────────────────────────────────────────────────────────────────────
    # HELPER: draw an insight text box below a chart panel
    # ─────────────────────────────────────────────────────────────────────────
    def insight_box(
        self, ax, text, facecolor="#F8F9FA", edgecolor="#3B6FD4", y=-0.35, fontsize=10
    ):
        """
        Draws a rounded text box below the given axis to display an insight.

        Parameters
        ----------
        ax         : Axes  – the subplot to attach the text to
        text       : str   – insight text to display
        facecolor  : str   – background color of the box
        edgecolor  : str   – border color of the box
        y          : float – vertical position in axes coords (default -0.24, below the axis)
        fontsize   : int   – font size of the insight text (default 10)

        Returns    : None  – draws directly onto ax
        """
        ax.text(
            0.01,
            y,
            text,
            transform=ax.transAxes,
            fontsize=17,
            color="#2A2D3E",
            bbox=dict(
                boxstyle="round,pad=0.8", facecolor=facecolor, edgecolor=edgecolor, lw=2
            ),
            fontweight="bold",
        )

    # ─────────────────────────────────────────────────────────────────────────
    # HELPER: apply common bar chart formatting (ticks, formatter, spines)
    # ─────────────────────────────────────────────────────────────────────────
    def format_yaxis_thousands(self, ax):
        """
        Applies a thousands formatter to the y-axis (e.g. 20000 → '20k').

        Parameters
        ----------
        ax  : Axes  – the subplot whose y-axis to format

        Returns : None  – modifies ax in place
        """
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(
                lambda v, _: f"{int(v / 1000)}k" if v >= 1000 else str(int(v))
            )
            # FuncFormatter: custom y-axis tick label function
            # lambda v, _: v=tick value, _=tick position (ignored)
            # if value ≥ 1000 → show as "20k"; otherwise show as plain integer
        )

    def kpi_card(self, ax, value, label, sub, accent, delta=None, delta_up=True):
        """
        Renders a KPI metric card on *ax*.

        Parameters
        ----------
        ax        : Axes   – subplot to draw on
        value     : str    – large headline number (e.g. "4.09")
        label     : str    – short descriptor under the number
        sub       : str    – small italic sub-text
        accent    : str    – hex colour used for the top bar and headline
        delta     : str    – optional trend line (e.g. "↑ 10× growth")
        delta_up  : bool   – True → green delta text; False → red

        Returns   : None   – draws directly onto ax, no return value
        """
        ax.set_facecolor(CARD_BG)  # white card background
        ax.set_xticks([])  # remove x-axis tick marks — cards have no data axis
        ax.set_yticks([])  # remove y-axis tick marks

        # ax.spines: dict of the 4 border lines (top/bottom/left/right)
        for sp in ax.spines.values():
            sp.set_edgecolor(BORDER_CLR)  # set border colour to light grey
            sp.set_linewidth(1)  # thin border line

        # Headline value — large bold number in accent color
        ax.text(
            # (x, y) in axes coordinates: 0.0=left/bottom, 1.0=right/top — centered horizontally, upper third
            0.5,  # x: centered horizontally
            0.66,  # y: upper third of the card
            value,  # the formatted string e.g. "96,096" or "R$137.75"
            # transAxes: coordinates are fractions of the axes (0–1), not data units
            transform=ax.transAxes,
            ha="center",  # horizontal alignment: center the text on x=0.5
            va="center",  # vertical alignment: center the text on y=0.66
            fontsize=24,  # large font to make the KPI number stand out
            fontweight="bold",  # bold weight for emphasis
            color=accent,  # use the passed accent color (blue/green/orange/red)
        )
        # Label — medium text describing the metric
        ax.text(
            0.5,  # x: centered horizontally
            0.45,  # y: slightly below the headline value
            label,  # e.g. "Total Orders"
            transform=ax.transAxes,  # axes fraction coordinates
            ha="center",  # center horizontally
            va="center",  # center vertically on y
            fontsize=15,  # medium font — secondary to the headline
            color=CANVAS_BG,  # medium grey — less prominent than the headline
            fontweight="bold",
        )
        # Sub-label — small italic context text
        ax.text(
            0.5,  # x: centered horizontally
            0.28,  # y: near the bottom of the card
            sub,  # e.g. "Unique order IDs"
            transform=ax.transAxes,  # axes fraction coordinates
            ha="center",  # center horizontally
            va="center",  # center vertically on y
            fontsize=13,  # small font — least prominent text
            color=TEXT_MUTED,  # light grey — least prominent text on the card
            fontweight="bold",
            # style="italic",  # italic to visually distinguish from the label
        )
        # Delta badge — optional trend indicator at the very bottom
        if delta:  # only draw if a delta string was passed
            ax.text(
                0.5,  # x: centered horizontally
                0.15,  # y: bottom of the card
                delta,  # e.g. "↑ 10× growth over dataset period"
                transform=ax.transAxes,  # axes fraction coordinates
                ha="center",  # center horizontally
                va="center",  # center vertically on y
                fontsize=10,  # small font for the delta badge
                color=(
                    # green for positive trends, red for negative
                    ACCENT_GREEN if delta_up else ACCENT_RED
                ),
                fontweight="bold",  # bold to make the trend indicator stand out
            )
            # Coloured top accent strip — a filled rectangle across the top of the card
        ax.add_patch(
            plt.Rectangle(
                # (x, y) bottom-left corner of the rectangle in axes coords — starts at 90% height
                (0, 0.90),
                1,  # width: full width of the card (100% in axes coords)
                0.10,  # height: 10% of the card height
                # use axes coords so it stays relative to the card size
                transform=ax.transAxes,
                color=accent,  # fill with the accent color (matches the headline number)
                # allow the strip to draw slightly outside the axes boundary if needed
                clip_on=False,
            )
        )

    def format_count(self, n):
        return f"{n / 1000:.1f}k" if n >= 1000 else str(n)

    def draw_dashboard_after_smote(self):
        # ── Figure Setup ──────────────────────────────────────────────────────────
        fig = plt.figure(figsize=(30, 45))
        fig.patch.set_facecolor(CANVAS_BG)

        # ── Custom Grid Layout ─────────────────────────────────────────────────────
        # Row 0 : 4 equal KPI cards
        # Row 1 : 1 wide main chart (all 4 cols)
        # Row 2 : price histogram (2 cols) | review bar (2 cols)
        # Row 3 : day-of-week bar (2 cols) | price-vs-freight scatter (2 cols)
        gs = gridspec.GridSpec(
            # nrows: 4 horizontal bands (KPIs / main chart / row2 / row3)
            6,
            # ncols: 4 vertical columns — KPI cards each take 1
            4,
            figure=fig,  # attach this grid to our figure
            hspace=0.9,  # vertical space between rows (fraction of avg row height)
            wspace=0.50,  # horizontal space between columns (fraction of avg col width)
            left=0.06,  # left margin of the entire grid (fraction of figure width)
            right=0.97,  # right margin of the entire grid
            top=0.95,  # top margin — leaves room for the main title above
            bottom=0.04,  # bottom margin — leaves room for insight text boxes below charts
            # relative heights per row: KPI cards are shorter, main chart is tallest
            height_ratios=[0.6, 4.0, 1.5, 1.5, 1.5, 1.5],
        )
        total_samples = len(self.full_df_transformed)
        avg_transformed_limit = self.full_df_transformed["LIMIT_BAL"].mean()
        target_counts = self.full_df_transformed[
            "default payment next month"
        ].value_counts()
        non_default_count = target_counts.get(0, 0)
        default_count = target_counts.get(1, 0)

        real_ratio_text = f"{self.format_count(non_default_count)} / {self.format_count(default_count)}"

        # ── 4. FIGURE SETUP ──────────────────────────────────────────────────────

        # ── 5. ROW 0: KPI CARDS ──────────────────────────────────────────────────
        self.kpi_card(
            fig.add_subplot(gs[0, 0]),
            f"{total_samples:,}",
            "Total Customers",
            "Post-SMOTE Balanced",
            ACCENT_BLUE,
        )
        self.kpi_card(
            fig.add_subplot(gs[0, 1]),
            real_ratio_text,
            "Non-Def / Default",
            "Post-SMOTE Balanced",
            ACCENT_RED,
        )
        self.kpi_card(
            fig.add_subplot(gs[0, 2]),
            f"{avg_transformed_limit:.2f}",
            "Avg Limit_Bal",
            "Power Transformed",
            ACCENT_GREEN,
        )
        self.kpi_card(
            fig.add_subplot(gs[0, 3]),
            "35",
            "Total Features",
            "Including Engineered",
            ACCENT_ORANGE,
        )

        # ── 6. ROW 1: MAIN HEATMAP (Full Width) ──────────────────────────────────
        ax_main = fig.add_subplot(gs[1, :])
        top_10_corr = (
            self.full_df_transformed.corr()["default payment next month"]
            .abs()
            .sort_values(ascending=False)
            .head(10)
            .index
        )
        sns.heatmap(
            self.full_df_transformed[top_10_corr].corr(),
            annot=True,
            fmt=".2f",
            cmap="Blues",
            center=0,
            ax=ax_main,
            annot_kws={"size": 16, "weight": "bold", "color": "#FFFFFF"},
            linewidths=0.5,
            linecolor="#E8E8F5",
        )
        ax_main.tick_params(axis="x", labelsize=17)
        ax_main.tick_params(axis="y", labelsize=17)
        ax_main.set_xticklabels(
            ax_main.get_xticklabels(),
            rotation=45,
            ha="right",
            fontsize=18,
            fontweight="bold",
            color="#FFFFFF",
        )
        ax_main.set_yticklabels(
            ax_main.get_yticklabels(),
            rotation=0,
            fontsize=18,
            fontweight="bold",
            color="#FFFFFF",
        )
        self.style_axis(ax_main, "Top 10 Features Correlation")
        self.insight_box(
            ax_main,
            """Delinquency-related features dominate the correlation — variables like rate_x_delinquency and unemp_x_delinquency show the strongest relationship with default (~0.40),
            confirming that past payment behavior is the primary driver of credit risk. SMOTE slightly amplifies this signal.""",
            facecolor="#F8F9FA",
            edgecolor=ACCENT_BLUE,
            y=-0.30,
        )

        # ── 7. ROW 2: CONTINUOUS DISTRIBUTIONS ──────────────────────────────────
        ax_avg_payment = fig.add_subplot(gs[2, 0:2])
        sns.kdeplot(
            data=self.full_df_transformed,
            x="avg_payment",
            hue="default payment next month",
            fill=True,
            ax=ax_avg_payment,
            palette=[ACCENT_BLUE, ACCENT_Pink],
        )
        self.style_axis(
            ax_avg_payment,
            "avg_payment by Default Status",
            ylabel="Density",
            xlabel="avg_payment",
        )
        self.insight_box(
            ax_avg_payment,
            """Lower avg_payment are realted with higher default risk,
            align with the weak negative correlations with PAY_AMT features (≈ -0.13 to -0.21).""",
            facecolor="#F8F9FA",
            edgecolor=ACCENT_BLUE,
        )

        ax_util = fig.add_subplot(gs[2, 2:4])
        sns.kdeplot(
            data=self.full_df_transformed,
            x="utilisation_ratio",
            hue="default payment next month",
            fill=True,
            ax=ax_util,
            palette=[ACCENT_BLUE, ACCENT_GREEN],
        )
        self.style_axis(
            ax_util,
            "Credit Utilisation by Default Status",
            ylabel="Density",
            xlabel="Utilisation Ratio",
        )
        self.insight_box(
            ax_util,
            "Higher credit utilisation is linked to increased default risk (positive correlation ≈ 0.14), indicating financial stress and over-reliance on credit.",
            facecolor="#F8F9FA",
            edgecolor=ACCENT_BLUE,
        )
        #############
        ax_limit = fig.add_subplot(gs[3, 0:2])
        ax_age = fig.add_subplot(gs[3, 2:4])

        sns.kdeplot(
            data=self.full_df_transformed,
            x="LIMIT_BAL",
            hue="default payment next month",
            fill=True,
            ax=ax_limit,
            palette=[ACCENT_BLUE, ACCENT_RED],
        )
        self.style_axis(
            ax_limit,
            "Transformed Limit Balance Density",
            ylabel="Density",
            xlabel="Transformed Limit",
        )
        self.insight_box(
            ax_limit,
            "Power Transformation fixed the skew...",
            facecolor="#F8F9FA",
            edgecolor=ACCENT_BLUE,
        )

        sns.kdeplot(
            data=self.full_df_transformed,
            x="AGE",
            hue="default payment next month",
            fill=True,
            ax=ax_age,
            palette=[ACCENT_BLUE, ACCENT_GREEN],
        )
        self.style_axis(
            ax_age,
            "Transformed Age Distribution",
            ylabel="Density",
            xlabel="Transformed Age",
        )
        self.insight_box(
            ax_age,
            "Power Transformation fixed the skew.",
            facecolor="#F8F9FA",
            edgecolor=ACCENT_RED,
        )
        # ── ROW 4: Education | Marriage ───────────────────────────────────────────
        ax_edu = fig.add_subplot(gs[4, 0])
        ax_mar = fig.add_subplot(gs[4, 1])
        ax_sex = fig.add_subplot(gs[4, 2])
        ax_is_underpaying = fig.add_subplot(gs[4, 3])

        # ── ROW 5 ─────────────────────────────────────────────────────────────────
        ax_pay = fig.add_subplot(gs[5, 0:2])
        ax_delinq = fig.add_subplot(gs[5, 2:4])

        # ── Education ─────────────────────────────────────────────────────────────
        ct_edu = pd.crosstab(
            self.full_df_transformed["EDUCATION"],
            self.full_df_transformed["default payment next month"],
            normalize="index",
        )
        ct_edu.plot(
            kind="bar",
            stacked=True,
            ax=ax_edu,
            color=[ACCENT_BLUE, ACCENT_RED],
            alpha=0.8,
        )
        self.style_axis(
            ax_edu,
            "Default by Education",
            ylabel="Proportion",
            xlabel="1=Graduate, 2=University, 3=High schools, 4=Other",
        )
        self.insight_box(
            ax_edu,
            "Education level 1,2 shows a higher default...",
            facecolor="#F8F9FA",
            edgecolor=ACCENT_ORANGE,
        )

        # ── Marriage ──────────────────────────────────────────────────────────────
        ct_mar = pd.crosstab(
            self.full_df_transformed["MARRIAGE"],
            self.full_df_transformed["default payment next month"],
            normalize="index",
        )
        ct_mar.plot(
            kind="bar",
            stacked=True,
            ax=ax_mar,
            color=[ACCENT_BLUE, ACCENT_RED],
            alpha=0.8,
        )
        self.style_axis(
            ax_mar,
            "Default by Marriage",
            ylabel="Proportion",
            xlabel="1=Married, 2=Single, 3=Other",
        )
        self.insight_box(
            ax_mar,
            "Marriage status shows a higher default...",
            facecolor="#F8F9FA",
            edgecolor=ACCENT_GREEN,
        )

        # ── SEX ───────────────────────────────────────────────────────────────────
        ct_sex = pd.crosstab(
            self.full_df_transformed["SEX"],
            self.full_df_transformed["default payment next month"],
            normalize="index",
        )
        ct_sex.plot(
            kind="bar",
            stacked=True,
            ax=ax_sex,
            color=[ACCENT_BLUE, ACCENT_RED],
            alpha=0.8,
        )
        self.style_axis(
            ax_sex, "Default by Gender", ylabel="Proportion", xlabel="1=Male, 2=Female"
        )
        self.insight_box(
            ax_sex,
            "Males shows a higher default...",
            facecolor="#F8F9FA",
            edgecolor=ACCENT_BLUE,
        )

        # ── is_underpaying ────────────────────────────────────────────────────────
        ct_under = pd.crosstab(
            self.full_df_transformed["is_underpaying"],
            self.full_df_transformed["default payment next month"],
            normalize="index",
        )
        ct_under.plot(
            kind="bar",
            stacked=True,
            ax=ax_is_underpaying,
            color=[ACCENT_BLUE, ACCENT_RED],
            alpha=0.8,
        )
        self.style_axis(
            ax_is_underpaying,
            "Default by Underpaying",
            ylabel="Proportion",
            xlabel="0=No, 1=Yes",
        )
        self.insight_box(
            ax_is_underpaying,
            """Underpaying shows minimal correlation with default, 
            suggesting it may not be a strong standalone predictor despite visible distribution differences.""",
            facecolor="#F8F9FA",
            edgecolor=ACCENT_RED,
        )
        # ── PAY_1 countplot ───────────────────────────────────────────────────────
        sns.countplot(
            data=self.full_df_transformed,
            x="PAY_1",
            hue="default payment next month",
            ax=ax_pay,
            palette=[ACCENT_BLUE, ACCENT_RED],
        )
        self.style_axis(
            ax_pay,
            "PAY_1 Distribution by Default",
            ylabel="Count",
            xlabel="PAY_1 Status",
        )
        self.insight_box(
            ax_pay,
            """Recent repayment status (PAY_1) shows the strongest behavioral signal (correlation ≈ 0.31),
            where delays of 2+ months sharply increase default risk.""",
            facecolor="#F8F9FA",
            edgecolor=ACCENT_BLUE,
        )

        # ── total_delinquency ─────────────────────────────────────────────────────
        bars = self.full_df_transformed.groupby("total_delinquency")[
            "default payment next month"
        ].mean()

        bars.plot(
            kind="bar",
            ax=ax_delinq,
            color=plt.cm.PiYG_r(range(0, 256, 256 // len(bars))),
            alpha=0.9,
        )
        self.style_axis(
            ax_delinq,
            "Default Rate by Total Delinquency",
            ylabel="Default Rate",
            xlabel="Total Delinquency Score",
        )
        self.insight_box(
            ax_delinq,
            "Higher delinquency score → sharply higher default rate.",
            facecolor="#F8F9FA",
            edgecolor=ACCENT_BLUE,
        )

        # ── 9. TITLE ─────────────────────────────────────────────────────────────
        fig.text(
            0.51,
            0.97,  #
            "Credit Risk Analysis Dashboard — (Post-SMOTE & Transformed)",
            ha="center",
            fontsize=26,
            fontweight="bold",
            color=TEXT_DARK,
        )
        plt.show()

    def draw_dashboard_before_smote(self):
        # # Add subplots at specific grid positions
        # ax_top_left   = fig.add_subplot(gs[0, 0])    # row 0, col 0
        # ax_top_right  = fig.add_subplot(gs[0, 1:3])  # row 0, cols 1–2 (wide panel!)
        # ax_full_width = fig.add_subplot(gs[1, :])    # row 1, all 3 cols
        # ── Color palette (light theme) ───────────────────────────────────────────
        ACCENT_BLUE = "#130A69"  # primary accent — orders, neutral info
        ACCENT_GREEN = "#2BA08B"  # positive / high ratings
        ACCENT_ORANGE = "#91734E"  # order value / weekends
        ACCENT_RED = "#690A42"  # negative / low ratings / delivery time
        ACCENT_Pink = "#CB7EE2"  # negative / low ratings / delivery time
        # CARD_BG = "#D0D0E2"  # dark card
        CANVAS_BG = "#22222C"  # darker background
        # BORDER_CLR = "#D8DCE8"  # card border colour
        TEXT_DARK = "#C2C8E9"  # headings
        # TEXT_MID = "#FFFFFF"  # labels
        # TEXT_MUTED = "#4A5068"  # sub-labels, axis ticks
        # GRID_CLR = "#EAECF4"  # chart grid lines

        # ── Figure Setup ──────────────────────────────────────────────────────────
        fig = plt.figure(figsize=(35, 45))
        fig.patch.set_facecolor(CANVAS_BG)

        # ── Custom Grid Layout ─────────────────────────────────────────────────────
        # Row 0 : 4 equal KPI cards
        # Row 1 : 1 wide main chart (all 4 cols)
        # Row 2 : price histogram (2 cols) | review bar (2 cols)
        # Row 3 : day-of-week bar (2 cols) | price-vs-freight scatter (2 cols)
        gs = gridspec.GridSpec(
            # nrows: 4 horizontal bands (KPIs / main chart / row2 / row3)
            6,
            # ncols: 4 vertical columns — KPI cards each take 1
            4,
            figure=fig,  # attach this grid to our figure
            hspace=0.9,  # vertical space between rows (fraction of avg row height)
            wspace=0.45,  # horizontal space between columns (fraction of avg col width)
            left=0.06,  # left margin of the entire grid (fraction of figure width)
            right=0.97,  # right margin of the entire grid
            top=0.95,  # top margin — leaves room for the main title above
            bottom=0.04,  # bottom margin — leaves room for insight text boxes below charts
            # relative heights per row: KPI cards are shorter, main chart is tallest
            height_ratios=[0.6, 4.0, 1.5, 1.5, 1.5, 1.5],
        )

        total_samples = len(self.full_df_transformed_without_smote)
        avg_transformed_limit = self.full_df_transformed_without_smote[
            "LIMIT_BAL"
        ].mean()
        target_counts = self.full_df_transformed_without_smote[
            "default payment next month"
        ].value_counts()
        non_default_count = target_counts.get(0, 0)
        default_count = target_counts.get(1, 0)

        real_ratio_text = f"{self.format_count(non_default_count)} / {self.format_count(default_count)}"

        # ── 4. FIGURE SETUP ──────────────────────────────────────────────────────

        # ── 5. ROW 0: KPI CARDS ──────────────────────────────────────────────────
        self.kpi_card(
            fig.add_subplot(gs[0, 0]),
            f"{total_samples:,}",
            "Total Customers",
            "Actuall number before SMOTE",
            ACCENT_BLUE,
        )
        self.kpi_card(
            fig.add_subplot(gs[0, 1]),
            real_ratio_text,
            "Non-Def / Default",
            "Actuall number before SMOT",
            ACCENT_RED,
        )
        self.kpi_card(
            fig.add_subplot(gs[0, 2]),
            f"{avg_transformed_limit:.2f}",
            "Avg Limit_Bal",
            "Power Transformed",
            ACCENT_GREEN,
        )
        self.kpi_card(
            fig.add_subplot(gs[0, 3]),
            "35",
            "Total Features",
            "Including Engineered",
            ACCENT_ORANGE,
        )

        # ── 6. ROW 1: MAIN HEATMAP (Full Width) ──────────────────────────────────
        ax_main = fig.add_subplot(gs[1, :])
        top_10_corr = (
            self.full_df_transformed_without_smote.corr()["default payment next month"]
            .abs()
            .sort_values(ascending=False)
            .head(10)
            .index
        )
        sns.heatmap(
            self.full_df_transformed_without_smote[top_10_corr].corr(),
            annot=True,
            fmt=".2f",
            cmap="Blues",
            center=0,
            ax=ax_main,
            annot_kws={"size": 16, "weight": "bold", "color": "#FFFFFF"},
            linewidths=0.5,
            linecolor="#E8E8F5",
        )
        ax_main.tick_params(axis="x", labelsize=17)
        ax_main.tick_params(axis="y", labelsize=17)
        ax_main.set_xticklabels(
            ax_main.get_xticklabels(),
            rotation=45,
            ha="right",
            fontsize=18,
            fontweight="bold",
            color="#FFFFFF",
        )
        ax_main.set_yticklabels(
            ax_main.get_yticklabels(),
            rotation=0,
            fontsize=18,
            fontweight="bold",
            color="#FFFFFF",
        )
        self.style_axis(ax_main, "Top 10 Features Correlation")
        self.insight_box(
            ax_main,
            "Total delinquency has a strong positive correlation with default (≈ 0.38), reinforcing that accumulated payment delays significantly increase risk.",
            facecolor="#F8F9FA",
            edgecolor=ACCENT_BLUE,
            y=-0.30,
        )

        # ── 7. ROW 2: CONTINUOUS DISTRIBUTIONS ──────────────────────────────────
        ax_avg_payment = fig.add_subplot(gs[2, 0:2])
        sns.kdeplot(
            data=self.full_df_transformed_without_smote,
            x="avg_payment",
            hue="default payment next month",
            fill=True,
            ax=ax_avg_payment,
            palette=[ACCENT_BLUE, ACCENT_Pink],
        )
        self.style_axis(
            ax_avg_payment,
            "avg_payment by Default Status",
            ylabel="Density",
            xlabel="avg_payment",
        )
        self.insight_box(
            ax_avg_payment,
            "As average payment amount decreases, the likelihood of default next month increases.",
            facecolor="#F8F9FA",
            edgecolor=ACCENT_BLUE,
        )

        ax_util = fig.add_subplot(gs[2, 2:4])
        sns.kdeplot(
            data=self.full_df_transformed_without_smote,
            x="utilisation_ratio",
            hue="default payment next month",
            fill=True,
            ax=ax_util,
            palette=[ACCENT_BLUE, ACCENT_GREEN],
        )
        self.style_axis(
            ax_util,
            "Credit Utilisation by Default Status",
            ylabel="Density",
            xlabel="Utilisation Ratio",
        )
        self.insight_box(
            ax_util,
            "Defaulters use a higher fraction of their credit limit — high utilisation signals financial stress.",
            facecolor="#F8F9FA",
            edgecolor=ACCENT_BLUE,
        )
        #############
        ax_limit = fig.add_subplot(gs[3, 0:2])
        ax_age = fig.add_subplot(gs[3, 2:4])

        sns.kdeplot(
            data=self.full_df_transformed_without_smote,
            x="LIMIT_BAL",
            hue="default payment next month",
            fill=True,
            ax=ax_limit,
            palette=[ACCENT_BLUE, ACCENT_RED],
        )
        self.style_axis(
            ax_limit,
            "Transformed Limit Balance Density",
            ylabel="Density",
            xlabel="Transformed Limit",
        )
        self.insight_box(
            ax_limit,
            "Power Transformation fixed the skew...",
            facecolor="#F8F9FA",
            edgecolor=ACCENT_BLUE,
        )

        sns.kdeplot(
            data=self.full_df_transformed_without_smote,
            x="AGE",
            hue="default payment next month",
            fill=True,
            ax=ax_age,
            palette=[ACCENT_BLUE, ACCENT_GREEN],
        )
        self.style_axis(
            ax_age,
            "Transformed Age Distribution",
            ylabel="Density",
            xlabel="Transformed Age",
        )
        self.insight_box(
            ax_age,
            "Power Transformation fixed the skew.",
            facecolor="#F8F9FA",
            edgecolor=ACCENT_RED,
        )
        # ── ROW 4: Education | Marriage ───────────────────────────────────────────
        ax_edu = fig.add_subplot(gs[4, 0])
        ax_mar = fig.add_subplot(gs[4, 1])
        ax_sex = fig.add_subplot(gs[4, 2])
        ax_is_underpaying = fig.add_subplot(gs[4, 3])

        # ── ROW 5 ─────────────────────────────────────────────────────────────────
        ax_pay = fig.add_subplot(gs[5, 0:2])
        ax_delinq = fig.add_subplot(gs[5, 2:4])

        # ── Education ─────────────────────────────────────────────────────────────
        ct_edu = pd.crosstab(
            self.full_df_transformed_without_smote["EDUCATION"],
            self.full_df_transformed_without_smote["default payment next month"],
            normalize="index",
        )
        ct_edu.plot(
            kind="bar",
            stacked=True,
            ax=ax_edu,
            color=[ACCENT_BLUE, ACCENT_RED],
            alpha=0.8,
        )
        self.style_axis(
            ax_edu,
            "Default by Education",
            ylabel="Proportion",
            xlabel="1=Graduate, 2=University, 3=High schools, 4=Other",
        )
        self.insight_box(
            ax_edu,
            "Education level 2,3 shows a higher default and are equal",
            facecolor="#F8F9FA",
            edgecolor=ACCENT_ORANGE,
        )

        # ── Marriage ──────────────────────────────────────────────────────────────
        ct_mar = pd.crosstab(
            self.full_df_transformed_without_smote["MARRIAGE"],
            self.full_df_transformed_without_smote["default payment next month"],
            normalize="index",
        )
        ct_mar.plot(
            kind="bar",
            stacked=True,
            ax=ax_mar,
            color=[ACCENT_BLUE, ACCENT_RED],
            alpha=0.8,
        )
        self.style_axis(
            ax_mar,
            "Default by Marriage",
            ylabel="Proportion",
            xlabel="1=Married, 2=Single, 3=Other",
        )
        self.insight_box(
            ax_mar,
            "Marriage status shows a equal default...",
            facecolor="#F8F9FA",
            edgecolor=ACCENT_GREEN,
        )

        # ── SEX ───────────────────────────────────────────────────────────────────
        ct_sex = pd.crosstab(
            self.full_df_transformed_without_smote["SEX"],
            self.full_df_transformed_without_smote["default payment next month"],
            normalize="index",
        )
        ct_sex.plot(
            kind="bar",
            stacked=True,
            ax=ax_sex,
            color=[ACCENT_BLUE, ACCENT_RED],
            alpha=0.8,
        )
        self.style_axis(
            ax_sex, "Default by Gender", ylabel="Proportion", xlabel="1=Male, 2=Female"
        )
        self.insight_box(
            ax_sex,
            "Gender shows a shows a equal default...",
            facecolor="#F8F9FA",
            edgecolor=ACCENT_BLUE,
        )

        # ── is_underpaying ────────────────────────────────────────────────────────
        ct_under = pd.crosstab(
            self.full_df_transformed_without_smote["is_underpaying"],
            self.full_df_transformed_without_smote["default payment next month"],
            normalize="index",
        )
        ct_under.plot(
            kind="bar",
            stacked=True,
            ax=ax_is_underpaying,
            color=[ACCENT_BLUE, ACCENT_RED],
            alpha=0.8,
        )
        self.style_axis(
            ax_is_underpaying,
            "Default by Underpaying",
            ylabel="Proportion",
            xlabel="0=No, 1=Yes",
        )
        self.insight_box(
            ax_is_underpaying,
            "Underpays default at a significantly higher rate.",
            facecolor="#F8F9FA",
            edgecolor=ACCENT_RED,
        )
        # ── PAY_1 countplot ───────────────────────────────────────────────────────
        sns.countplot(
            data=self.full_df_transformed_without_smote,
            x="PAY_1",
            hue="default payment next month",
            ax=ax_pay,
            palette=[ACCENT_BLUE, ACCENT_RED],
        )
        self.style_axis(
            ax_pay,
            "PAY_1 Distribution by Default",
            ylabel="Count",
            xlabel="PAY_1 Status",
        )
        self.insight_box(
            ax_pay,
            "PAY_1 = 2+ means 2+ months delayed — default rate spikes sharply.",
            facecolor="#F8F9FA",
            edgecolor=ACCENT_BLUE,
        )

        # ── total_delinquency ─────────────────────────────────────────────────────
        bars = self.full_df_transformed_without_smote.groupby("total_delinquency")[
            "default payment next month"
        ].mean()

        bars.plot(
            kind="bar",
            ax=ax_delinq,
            color=plt.cm.PiYG_r(range(0, 256, 256 // len(bars))),
            alpha=0.9,
        )
        self.style_axis(
            ax_delinq,
            "Default Rate by Total Delinquency",
            ylabel="Default Rate",
            xlabel="Total Delinquency Score",
        )
        self.insight_box(
            ax_delinq,
            "Higher delinquency score → customers with delayed bills increses.",
            facecolor="#F8F9FA",
            edgecolor=ACCENT_BLUE,
        )

        # ── 9. TITLE ─────────────────────────────────────────────────────────────
        fig.text(
            0.51,
            0.97,  #
            "Credit Risk Analysis Dashboard — (Before-SMOTE & Transformed)",
            ha="center",
            fontsize=26,
            fontweight="bold",
            color=TEXT_DARK,
        )
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
