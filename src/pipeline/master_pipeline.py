import argparse
import logging
import sys
import time


from src.pipeline.data_acquisition import run_acquisition  # Phase 1
from src.pipeline.data_validation import run_validation  # Phase 2
from src.pipeline.eda import run_eda
from src.pipeline.data_cleaning import clean_data  # Phase 3  <- uncomment when ready
from src.pipeline.data_transformation import (
    run_transformation,
)  # Phase 4  <- uncomment when ready
# from src.pipeline.model_training       import run_training        # Phase 5  <- uncomment when ready
# from src.pipeline.model_evaluation     import run_evaluation      # Phase 6  <- uncomment when ready

# Logging
# ===========================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)

# Phase registry — add a dict entry for every completed phase
# ===========================================================================

_PHASES: list[dict] = [
    # Phase 1 — Data Acquisition
    {
        "name": "1 | Data Acquisition",
        "fn": run_acquisition,
        "kwargs": {},
    },
    # Phase 2 — Data Validation
    # -------------------------------------------------------------------------
    {
        "name": "2 | Data Validation",
        "fn": run_validation,
        "kwargs": {
            "merged_path": "data/taiwan_merged.csv",
            "json_output": "data/validation_results.json",
            "md_output": "data/validation_report.md",
        },
    },
    # Phase 3 — Data Cleaning              (add when ready)
    {
        "name": "3 | Data Cleaning",
        "fn": clean_data,
        "kwargs": {
            "input_path": "data/taiwan_merged.csv",
            "train_output": "data/clean/train_cleaned.csv",
            "val_output": "data/clean/val_cleaned.csv",
            "test_output": "data/clean/test_cleaned.csv",
        },
    },
    {
        "name": "4 | Data Transformation",
        "fn": run_transformation,
        "kwargs": {
            "train_input": "data/clean/train_cleaned.csv",
            "val_input": "data/clean/val_cleaned.csv",
            "test_input": "data/clean/test_cleaned.csv",
            "train_output": "data/transformed/train_transformed.csv",
            "val_output": "data/transformed/val_transformed.csv",
            "test_output": "data/transformed/test_transformed.csv",
        },
    },
    # Phase 5 — EDA
    {
        "name": "5 | EDA",
        "fn": run_eda,
        "kwargs": {
            "train_input_cleaned": "data/clean/train_cleaned.csv",
            "val_input_cleaned": "data/clean/val_cleaned.csv",
            "test_input_cleaned": "data/clean/test_cleaned.csv",
            "train_input_transformed": "data/transformed/train_transformed.csv",
            "val_input_transformed": "data/transformed/val_transformed.csv",
            "test_input_transformed": "data/transformed/test_transformed.csv",
        },
    },
    # Phase 5 — Model Training             (add when ready)
    # Phase 6 — Model Evaluation           (add when ready)
]


def run_pipeline(start_from: int = 1) -> None:
    """
    Execute every registered phase in order
    """
    phases = [p for p in _PHASES if _phase_number(p["name"]) >= start_from]

    if not phases:
        log.error(
            "No phases match --from %d. Available phases: 1-%d",
            start_from,
            len(_PHASES),
        )
        sys.exit(1)

    log.info("=" * 60)
    log.info("INTEGRATION PIPELINE — %d phase(s) queued", len(phases))
    if start_from > 1:
        log.info("Resuming from Phase %d", start_from)
    log.info("=" * 60)

    pipeline_start = time.time()
    failed_phase = None

    for _, phase in enumerate(phases, start=1):
        log.info("")
        log.info("  PHASE %s", phase["name"])
        log.info("-" * 60)
        step_start = time.time()

        try:
            phase["fn"](**phase["kwargs"])
            elapsed = time.time() - step_start
            log.info("  PHASE %s — completed in %.1fs", phase["name"], elapsed)

        except Exception as exc:
            elapsed = time.time() - step_start
            log.error("  PHASE %s — FAILED after %.1fs", phase["name"], elapsed)
            log.error("    %s: %s", type(exc).__name__, exc)
            failed_phase = phase["name"]
            break

    total = time.time() - pipeline_start
    log.info("")
    log.info("=" * 60)

    if failed_phase:
        log.error("PIPELINE STOPPED at %s (%.1fs total)", failed_phase, total)
        log.error(
            "Fix the error above, then resume with:  --from %s",
            _phase_number(failed_phase),
        )
        sys.exit(1)
    else:
        log.info("PIPELINE COMPLETE — all phases passed (%.1fs total)", total)
        log.info("=" * 60)


def _phase_number(name: str) -> int:
    try:
        return int(name.split("|")[0].strip())
    except (ValueError, IndexError):
        return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the full data-science integration pipeline"
    )
    parser.add_argument(
        "--from",
        dest="start_from",
        type=int,
        default=1,
        metavar="N",
        help="Start from phase N (default: 1). Useful for resuming after a failure",
    )
    args = parser.parse_args()
    run_pipeline(start_from=args.start_from)
