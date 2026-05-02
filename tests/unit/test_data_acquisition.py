import pytest
import sys
import os
from pathlib import Path
from unittest.mock import patch, MagicMock, call

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)


class TestStepHelper:

    def test_returns_function_result(self):
        from src.pipeline.data_acquisition import _step

        result = _step("test", lambda: 42)
        assert result == 42

    def test_passes_args_and_kwargs(self):
        from src.pipeline.data_acquisition import _step

        fn = MagicMock(return_value="ok")
        _step("test", fn, "arg1", kw="val")
        fn.assert_called_once_with("arg1", kw="val")

    def test_wraps_exception_in_runtime_error(self):
        from src.pipeline.data_acquisition import _step

        def boom():
            raise ValueError("inner error")

        with pytest.raises(RuntimeError, match="inner error"):
            _step("failing step", boom)

    def test_runtime_error_message_contains_step_name(self):
        from src.pipeline.data_acquisition import _step

        with pytest.raises(RuntimeError, match="my step"):
            _step("my step", lambda: (_ for _ in ()).throw(OSError("oops")))

    def test_original_exception_chained(self):
        from src.pipeline.data_acquisition import _step

        try:
            _step("s", lambda: (_ for _ in ()).throw(KeyError("k")))
        except RuntimeError as exc:
            assert isinstance(exc.__cause__, KeyError)


class TestRunAcquisitionGuards:
    def test_missing_api_key_raises_environment_error(self, tmp_path):
        from src.pipeline.data_acquisition import run_acquisition

        taiwan = tmp_path / "taiwan.xls"
        taiwan.write_bytes(b"dummy")

        with patch.dict(os.environ, {}, clear=True):
            with patch("src.pipeline.data_acquisition.load_dotenv"):
                with pytest.raises(EnvironmentError, match="FRED_API_KEY"):
                    run_acquisition(
                        taiwan_path=str(taiwan),
                        fred_api_key=None,
                    )

    def test_missing_taiwan_file_raises_file_not_found(self, tmp_path):
        from src.pipeline.data_acquisition import run_acquisition

        with pytest.raises(FileNotFoundError, match="Taiwan dataset not found"):
            run_acquisition(
                taiwan_path=str(tmp_path / "nonexistent.xls"),
                fred_api_key="fake-key",
            )

    def test_explicit_api_key_bypasses_env_lookup(self, tmp_path):
        from src.pipeline.data_acquisition import run_acquisition

        taiwan = tmp_path / "taiwan.xls"
        taiwan.write_bytes(b"dummy")

        # All external calls mocked — we just want no EnvironmentError
        with patch("src.pipeline.data_acquisition.fetch_macro_data"), patch(
            "src.pipeline.data_acquisition.scrape_taiex"
        ), patch("src.pipeline.data_acquisition.parse_cbc_rates"), patch(
            "src.pipeline.data_acquisition.scrape_unemployment"
        ), patch(
            "src.pipeline.data_acquisition.build_final_dataset"
        ), patch(
            "src.pipeline.data_acquisition.load_dotenv"
        ), patch.dict(
            os.environ, {}, clear=True
        ):
            run_acquisition(
                taiwan_path=str(taiwan),
                macro_path=str(tmp_path / "macro.csv"),
                taiex_path=str(tmp_path / "taiex.csv"),
                cbc_path=str(tmp_path / "cbc.csv"),
                unemploy_path=str(tmp_path / "unemploy.csv"),
                merged_path=str(tmp_path / "merged.csv"),
                fred_api_key="explicit-key",
            )


class TestRunAcquisitionOrchestration:

    @pytest.fixture()
    def patched_run(self, tmp_path):
        taiwan = tmp_path / "taiwan.xls"
        taiwan.write_bytes(b"dummy")

        macro_path = str(tmp_path / "macro.csv")
        taiex_path = str(tmp_path / "taiex.csv")
        cbc_path = str(tmp_path / "cbc.csv")
        unemploy_path = str(tmp_path / "unemploy.csv")
        merged_path = str(tmp_path / "merged.csv")

        mocks = {}
        with patch("src.pipeline.data_acquisition.fetch_macro_data") as m1, patch(
            "src.pipeline.data_acquisition.scrape_taiex"
        ) as m2, patch("src.pipeline.data_acquisition.parse_cbc_rates") as m3, patch(
            "src.pipeline.data_acquisition.scrape_unemployment"
        ) as m4, patch(
            "src.pipeline.data_acquisition.build_final_dataset"
        ) as m5, patch(
            "src.pipeline.data_acquisition.load_dotenv"
        ):
            mocks = {
                "fetch_macro_data": m1,
                "scrape_taiex": m2,
                "parse_cbc_rates": m3,
                "scrape_unemployment": m4,
                "build_final_dataset": m5,
            }

            from src.pipeline.data_acquisition import run_acquisition

            run_acquisition(
                taiwan_path=str(taiwan),
                macro_path=macro_path,
                taiex_path=taiex_path,
                cbc_path=cbc_path,
                unemploy_path=unemploy_path,
                merged_path=merged_path,
                fred_api_key="test-key",
            )

        yield mocks, {
            "taiwan": str(taiwan),
            "macro": macro_path,
            "taiex": taiex_path,
            "cbc": cbc_path,
            "unemploy": unemploy_path,
            "merged": merged_path,
        }

    def test_fetch_macro_data_called_once(self, patched_run):
        mocks, _ = patched_run
        mocks["fetch_macro_data"].assert_called_once()

    def test_fetch_macro_data_receives_api_key(self, patched_run):
        mocks, paths = patched_run
        _, kwargs = mocks["fetch_macro_data"].call_args
        assert kwargs.get("api_key") == "test-key"

    def test_fetch_macro_data_receives_save_path(self, patched_run):
        mocks, paths = patched_run
        _, kwargs = mocks["fetch_macro_data"].call_args
        assert kwargs.get("save_path") == paths["macro"]

    def test_scrape_taiex_called_once(self, patched_run):
        mocks, _ = patched_run
        mocks["scrape_taiex"].assert_called_once()

    def test_scrape_taiex_receives_save_path(self, patched_run):
        mocks, paths = patched_run
        _, kwargs = mocks["scrape_taiex"].call_args
        assert kwargs.get("save_path") == paths["taiex"]

    def test_parse_cbc_rates_called_once(self, patched_run):
        mocks, _ = patched_run
        mocks["parse_cbc_rates"].assert_called_once()

    def test_parse_cbc_rates_receives_save_path(self, patched_run):
        mocks, paths = patched_run
        _, kwargs = mocks["parse_cbc_rates"].call_args
        assert kwargs.get("save_path") == paths["cbc"]

    def test_scrape_unemployment_called_once(self, patched_run):
        mocks, _ = patched_run
        mocks["scrape_unemployment"].assert_called_once()

    def test_scrape_unemployment_receives_save_path(self, patched_run):
        mocks, paths = patched_run
        _, kwargs = mocks["scrape_unemployment"].call_args
        assert kwargs.get("save_path") == paths["unemploy"]

    def test_build_final_dataset_called_once(self, patched_run):
        mocks, _ = patched_run
        mocks["build_final_dataset"].assert_called_once()

    def test_build_final_dataset_receives_all_paths(self, patched_run):
        mocks, paths = patched_run
        _, kwargs = mocks["build_final_dataset"].call_args
        assert kwargs["taiwan_path"] == paths["taiwan"]
        assert kwargs["macro_path"] == paths["macro"]
        assert kwargs["taiex_path"] == paths["taiex"]
        assert kwargs["cbc_path"] == paths["cbc"]
        assert kwargs["unemploy_path"] == paths["unemploy"]
        assert kwargs["save_path"] == paths["merged"]

    def test_step_failure_propagates_as_runtime_error(self, tmp_path):
        from src.pipeline.data_acquisition import run_acquisition

        taiwan = tmp_path / "taiwan.xls"
        taiwan.write_bytes(b"dummy")

        with patch(
            "src.pipeline.data_acquisition.fetch_macro_data",
            side_effect=ConnectionError("API down"),
        ), patch("src.pipeline.data_acquisition.load_dotenv"):
            with pytest.raises(RuntimeError, match="API down"):
                run_acquisition(
                    taiwan_path=str(taiwan),
                    fred_api_key="key",
                )

    def test_all_five_steps_called_in_sequence(self, patched_run):
        mocks, _ = patched_run
        for name, mock in mocks.items():
            assert mock.call_count == 1, f"{name} was not called exactly once"
