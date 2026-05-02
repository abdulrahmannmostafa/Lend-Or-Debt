import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch

N = 10  # number of synthetic borrowers


def _make_taiwan_df(n=N) -> pd.DataFrame:
    data = {
        "ID": range(1, n + 1),
        "LIMIT_BAL": [50_000 * (i + 1) for i in range(n)],
        "SEX": [1, 2] * (n // 2),
        "EDUCATION": [1, 2, 3, 4] * (n // 4) + [1, 2],
        "MARRIAGE": [1, 2, 3] * (n // 3) + [1],
        "AGE": [25 + i for i in range(n)],
        # PAY_STATUS columns
        "PAY_0": [0] * n,
        "PAY_2": [0] * n,
        "PAY_3": [-1] * n,
        "PAY_4": [-1] * n,
        "PAY_5": [-2] * n,
        "PAY_6": [-2] * n,
        # BILL amounts
        **{f"BILL_AMT{i}": [1000 * i] * n for i in range(1, 7)},
        # PAY amounts
        **{f"PAY_AMT{i}": [500 * i] * n for i in range(1, 7)},
        "default payment next month": [0, 1] * (n // 2),
    }
    return pd.DataFrame(data)


def _make_macro_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Year": [2005] * 6,
            "Month": [4, 5, 6, 7, 8, 9],
            "CPI": [104.1, 104.5, 104.8, 105.0, 105.3, 105.5],
            "GDP": [12500.0] * 6,
        }
    )


def _make_taiex_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Year": [2005] * 6,
            "Month": [4, 5, 6, 7, 8, 9],
            "TAIEX_close": [6100.0, 6200.0, 6050.0, 6300.0, 6400.0, 6500.0],
        }
    )


def _make_cbc_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Year": [2005] * 6,
            "Month": [4, 5, 6, 7, 8, 9],
            "Discount_Rate": [2.000, 2.000, 2.000, 2.125, 2.125, 2.125],
        }
    )


def _make_unemploy_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Year": [2005] * 6,
            "Month": [4, 5, 6, 7, 8, 9],
            "Unemployment_Rate": [4.13, 4.13, 4.13, 4.05, 4.05, 4.05],
        }
    )


class TestBuildPanel:
    def test_row_count(self):
        from src.scrape.merge_sources import _build_panel

        taiwan = _make_taiwan_df()
        panel = _build_panel(taiwan)
        assert len(panel) == N * 6, "Expected N borrowers × 6 months"

    def test_correct_months_present(self):
        from src.scrape.merge_sources import _build_panel

        panel = _build_panel(_make_taiwan_df())
        assert set(panel["Month"]) == {4, 5, 6, 7, 8, 9}

    def test_year_is_2005(self):
        from src.scrape.merge_sources import _build_panel

        panel = _build_panel(_make_taiwan_df())
        assert (panel["Year"] == 2005).all()

    def test_required_columns_present(self):
        from src.scrape.merge_sources import _build_panel

        panel = _build_panel(_make_taiwan_df())
        for col in ["ID", "Year", "Month", "PAY_STATUS", "BILL_AMT", "PAY_AMT"]:
            assert col in panel.columns

    def test_each_borrower_has_6_rows(self):
        from src.scrape.merge_sources import _build_panel

        panel = _build_panel(_make_taiwan_df())
        counts = panel.groupby("ID").size()
        assert (counts == 6).all()

    def test_september_maps_to_pay_0(self):
        """The MONTH_PANEL mapping says September (Month=9) -> PAY_0."""
        from src.scrape.merge_sources import _build_panel

        taiwan = _make_taiwan_df()
        taiwan["PAY_0"] = 3  # set a distinctive value
        panel = _build_panel(taiwan)
        sept_rows = panel[panel["Month"] == 9]
        assert (sept_rows["PAY_STATUS"] == 3).all()


class TestRecodePayStatus:
    def test_negative_values_clipped_to_zero(self):
        from src.scrape.merge_sources import _recode_pay_status

        df = pd.DataFrame({"PAY_STATUS": [-2, -1, 0, 1, 2, 8]})
        out = _recode_pay_status(df)
        assert out["PAY_STATUS_clean"].min() == 0

    def test_positive_values_unchanged(self):
        from src.scrape.merge_sources import _recode_pay_status

        df = pd.DataFrame({"PAY_STATUS": [1, 2, 8]})
        out = _recode_pay_status(df)
        assert list(out["PAY_STATUS_clean"]) == [1, 2, 8]

    def test_original_column_preserved(self):
        from src.scrape.merge_sources import _recode_pay_status

        df = pd.DataFrame({"PAY_STATUS": [-2, 0, 3]})
        out = _recode_pay_status(df)
        assert "PAY_STATUS" in out.columns
        assert list(out["PAY_STATUS"]) == [-2, 0, 3]


class TestAggregateBehaviour:
    def _make_panel(self):
        from src.scrape.merge_sources import _build_panel, _recode_pay_status

        taiwan = _make_taiwan_df()
        panel = _build_panel(taiwan)
        panel = panel.merge(_make_macro_df(), on=["Year", "Month"], how="left")
        panel = panel.merge(
            _make_taiex_df()[["Year", "Month", "TAIEX_close"]],
            on=["Year", "Month"],
            how="left",
        )
        panel = panel.merge(
            _make_cbc_df()[["Year", "Month", "Discount_Rate"]],
            on=["Year", "Month"],
            how="left",
        )
        panel = panel.merge(
            _make_unemploy_df()[["Year", "Month", "Unemployment_Rate"]],
            on=["Year", "Month"],
            how="left",
        )
        return _recode_pay_status(panel)

    def test_one_row_per_borrower(self):
        from src.scrape.merge_sources import _aggregate_behaviour

        agg = _aggregate_behaviour(self._make_panel())
        assert len(agg) == N

    def test_required_agg_columns(self):
        from src.scrape.merge_sources import _aggregate_behaviour

        agg = _aggregate_behaviour(self._make_panel())
        for col in [
            "avg_bill",
            "avg_payment",
            "max_delinquency",
            "total_delinquency",
            "avg_macro_CPI",
            "avg_macro_GDP",
            "avg_macro_TAIEX",
            "avg_macro_rate",
            "avg_macro_unemp",
        ]:
            assert col in agg.columns

    def test_max_delinquency_gte_zero(self):
        from src.scrape.merge_sources import _aggregate_behaviour

        agg = _aggregate_behaviour(self._make_panel())
        assert (agg["max_delinquency"] >= 0).all()

    def test_avg_payment_gte_zero(self):
        from src.scrape.merge_sources import _aggregate_behaviour

        agg = _aggregate_behaviour(self._make_panel())
        assert (agg["avg_payment"] >= 0).all()


class TestBuildFinalDataset:
    @pytest.fixture()
    def mock_loaders(self, tmp_path):
        taiwan = _make_taiwan_df()
        macro = _make_macro_df()
        taiex = _make_taiex_df()
        cbc = _make_cbc_df()
        unemp = _make_unemploy_df()

        save_path = str(tmp_path / "merged.csv")

        with patch(
            "src.scrape.merge_sources.pd.read_excel", return_value=taiwan
        ), patch(
            "src.scrape.merge_sources.pd.read_csv",
            side_effect=[macro, taiex, cbc, unemp],
        ):
            from src.scrape.merge_sources import build_final_dataset

            result = build_final_dataset(
                taiwan_path="fake.xls",
                macro_path="fake_macro.csv",
                taiex_path="fake_taiex.csv",
                cbc_path="fake_cbc.csv",
                unemploy_path="fake_unemp.csv",
                save_path=save_path,
            )
        yield result, save_path

    def test_returns_dataframe(self, mock_loaders):
        result, _ = mock_loaders
        assert isinstance(result, pd.DataFrame)

    def test_row_count_matches_borrowers(self, mock_loaders):
        result, _ = mock_loaders
        assert len(result) == N, "One row per borrower — no merge fan-out"

    def test_id_column_dropped(self, mock_loaders):
        result, _ = mock_loaders
        assert "ID" not in result.columns

    def test_no_nulls_in_macro_columns(self, mock_loaders):
        result, _ = mock_loaders
        macro_cols = [
            "avg_macro_CPI",
            "avg_macro_GDP",
            "avg_macro_TAIEX",
            "avg_macro_rate",
            "avg_macro_unemp",
        ]
        for col in macro_cols:
            assert result[col].notna().all(), f"Null found in {col}"

    def test_target_column_preserved(self, mock_loaders):
        result, _ = mock_loaders
        assert "default payment next month" in result.columns
        assert result["default payment next month"].isin({0, 1}).all()

    def test_csv_is_written(self, mock_loaders):
        _, path = mock_loaders
        import os

        assert os.path.exists(path)

    def test_no_merge_fan_out(self, mock_loaders):
        result, _ = mock_loaders
        assert len(result) == N
