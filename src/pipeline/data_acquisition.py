import logging
import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

from src.scrape.fetch_fred_api import fetch_macro_data
from src.scrape.merge_sources import build_final_dataset
from src.scrape.parse_cbc_pdf import parse_cbc_rates
from src.scrape.scrape_taiex import scrape_taiex
from src.scrape.scrape_unemployment import scrape_unemployment

_THIS_DIR = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _THIS_DIR.parent
_DATA_DIR = _PROJECT_ROOT / "data"
sys.path.insert(0, str(_THIS_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)


DEFAULT_PATHS = {
    "taiwan": str(_DATA_DIR / "default of credit card clients.xls"),
    "macro": str(_DATA_DIR / "macro_fred.csv"),
    "taiex": str(_DATA_DIR / "taiex_2005.csv"),
    "cbc": str(_DATA_DIR / "cbc_rates_2005.csv"),
    "unemploy": str(_DATA_DIR / "unemployment_2005.csv"),
    "merged": str(_DATA_DIR / "taiwan_merged.csv"),
}


# A wrapper function that runs any pipeline step with consistent logging and error handling
def _step(name: str, fn, *args, **kwargs):
    log.info("=" * 60)
    log.info("STEP: %s", name)
    log.info("=" * 60)
    start = time.time()
    try:
        result = fn(*args, **kwargs)
        elapsed = time.time() - start
        log.info("DONE: %s (%.1fs)", name, elapsed)
        return result
    except Exception as exc:
        log.error("FAILED: %s — %s: %s", name, type(exc).__name__, exc)
        raise RuntimeError(f"Acquisition step '{name}' failed: {exc}") from exc


def run_acquisition(
    taiwan_path: str = DEFAULT_PATHS["taiwan"],
    macro_path: str = DEFAULT_PATHS["macro"],
    taiex_path: str = DEFAULT_PATHS["taiex"],
    cbc_path: str = DEFAULT_PATHS["cbc"],
    unemploy_path: str = DEFAULT_PATHS["unemploy"],
    merged_path: str = DEFAULT_PATHS["merged"],
    fred_api_key: str | None = None,
) -> None:
    load_dotenv()

    api_key = fred_api_key or os.getenv("FRED_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "FRED_API_KEY is not set "
            "Add it to your .env file or pass fred_api_key= explicitly"
        )

    if not Path(taiwan_path).exists():
        raise FileNotFoundError(
            f"Taiwan dataset not found: {taiwan_path}\n"
            "Download it from: https://archive.ics.uci.edu/dataset/350/"
            "default+of+credit+card+clients and place it in data/."
        )

    log.info("Starting data acquisition pipeline")
    log.info("Taiwan source : %s", taiwan_path)
    log.info("Output dir    : %s", _DATA_DIR)
    pipeline_start = time.time()

    _step(
        "FRED API — CPI + GDP (2005)",
        fetch_macro_data,
        api_key=api_key,
        save_path=macro_path,
    )

    _step(
        "Yahoo Finance — TAIEX (Apr-Sep 2005)",
        scrape_taiex,
        save_path=taiex_path,
    )

    _step(
        "CBC Annual Report — Discount Rate (Apr-Sep 2005)",
        parse_cbc_rates,
        save_path=cbc_path,
    )

    _step(
        "DGBAS XML — Unemployment Rate (Apr-Sep 2005)",
        scrape_unemployment,
        save_path=unemploy_path,
    )

    _step(
        "Merge — Taiwan's Dataset + 4 macro sources -> taiwan_merged.csv",
        build_final_dataset,
        taiwan_path=taiwan_path,
        macro_path=macro_path,
        taiex_path=taiex_path,
        cbc_path=cbc_path,
        unemploy_path=unemploy_path,
        save_path=merged_path,
    )

    total = time.time() - pipeline_start
    log.info("=" * 60)
    log.info("DATA ACQUISITION COMPLETE (%.1fs total)", total)
    log.info("Outputs:")
    for label, path in [
        ("FRED macro", macro_path),
        ("TAIEX", taiex_path),
        ("CBC rates", cbc_path),
        ("Unemployment", unemploy_path),
        ("Merged dataset", merged_path),
    ]:
        size = Path(path).stat().st_size / 1024 if Path(path).exists() else 0
        log.info("  %-20s -> %s (%.1f KB)", label, path, size)
    log.info("=" * 60)
    log.info("Next step: run src/pipeline/data_validation.py")


if __name__ == "__main__":
    run_acquisition()
