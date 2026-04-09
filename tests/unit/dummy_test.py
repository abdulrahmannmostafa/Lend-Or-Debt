from src.pipeline.data_validation import run_validation


def test_import():
    assert run_validation is not None
