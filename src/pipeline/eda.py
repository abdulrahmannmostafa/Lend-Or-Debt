import logging
import pandas as pd  # Data manipulation and analysis
import numpy as np  # Numerical computing
import matplotlib.pyplot as plt  # Core plotting (the matplotlib.pyplot module)
import seaborn as sns  # Statistical visualizations (built on matplotlib)
from pathlib import Path

plt.style.use("dark_background")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)


class EDA:
    def __init__(
        self,
        train_input_cleaned,
        val_input_cleaned,
        test_input_cleaned,
        train_input_transformed,
        val_input_transformed,
        test_input_transformed,
    ):
        log.info("EDA Module has started")
        self.full_df_transformed = None
        self.full_df_cleaned = None
        self.train_input_cleaned = train_input_cleaned
        self.val_input_cleaned = val_input_cleaned
        self.test_input_cleaned = test_input_cleaned
        self.train_input_transformed = train_input_transformed
        self.val_input_transformed = val_input_transformed
        self.test_input_transformed = test_input_transformed

    def load_data_transformed(self):
        log.info("This loader for transformed data eda")
        log.info("train data is loaded")
        train_dataset = pd.read_csv(self.train_input_transformed)
        log.info("val data is loaded")
        val_dataset = pd.read_csv(self.val_input_transformed)
        log.info("test data is loaded")
        test_dataset = pd.read_csv(self.test_input_transformed)

        print(f"train shape is: {train_dataset.shape}")
        print(f"val shape is: {val_dataset.shape}")
        print(f"test shape is: {test_dataset.shape}")
        full_df = pd.concat(
            [train_dataset, val_dataset, test_dataset], ignore_index=True
        )
        # ignore_index=True --> to combine index from 0 to N  instead of duplicates
        self.full_df_transformed = full_df
        log.info("full data has being created and ready for EDA")
        print(f"full data shape is: {full_df.shape}")
        return train_dataset, val_dataset, test_dataset, full_df

    def load_data_cleaned(self):
        log.info("This loader for cleaned data eda")
        log.info("train data is loaded")
        train_dataset = pd.read_csv(self.train_input_cleaned)
        log.info("val data is loaded")
        val_dataset = pd.read_csv(self.val_input_cleaned)
        log.info("test data is loaded")
        test_dataset = pd.read_csv(self.test_input_cleaned)

        print(f"train shape is: {train_dataset.shape}")
        print(f"val shape is: {val_dataset.shape}")
        print(f"test shape is: {test_dataset.shape}")
        full_df = pd.concat(
            [train_dataset, val_dataset, test_dataset], ignore_index=True
        )
        # ignore_index=True --> to combine index from 0 to N  instead of duplicates
        self.full_df_cleaned = full_df
        log.info("full data has being created and ready for EDA")
        print(f"full data shape is: {full_df.shape}")
        return train_dataset, val_dataset, test_dataset, full_df

    def apply_univariate(self, column_name, data_type=0):
        # ─── 1. UNIVARIATE: Distribution of Limit Bal ──────────────────────────
        # WHY: Price is right-skewed (few very expensive orders).
        # Histogram + KDE together show both frequency and shape.
        if data_type:
            full_df = self.full_df_cleaned.copy()
        else:
            full_df = self.full_df_transformed.copy()
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
        if data_type:
            full_df = self.full_df_cleaned.copy()
        else:
            full_df = self.full_df_transformed.copy()
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
        log.warning(f"Auto-eda failed during import: {e}")
