import pytest
import sys
import os
from unittest.mock import patch, MagicMock, call

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)


def _make_phase(number: int, mock_fn=None) -> dict:
    """Build a minimal phase dict as expected by run_pipeline."""
    return {
        "name": f"{number} | Phase {number}",
        "fn": mock_fn or MagicMock(return_value=None),
        "kwargs": {"arg": f"val{number}"},
    }


class TestPhaseNumber:
    def test_extracts_integer_from_name(self):
        from src.pipeline.master_pipeline import _phase_number

        assert _phase_number("3 | Data Cleaning") == 3

    def test_extracts_multi_digit_number(self):
        from src.pipeline.master_pipeline import _phase_number

        assert _phase_number("12 | Something") == 12

    def test_returns_zero_for_bad_format(self):
        from src.pipeline.master_pipeline import _phase_number

        assert _phase_number("no number here") == 0

    def test_works_with_real_phase_names(self):
        from src.pipeline.master_pipeline import _phase_number

        names = [
            "1 | Data Acquisition",
            "2 | Data Validation",
            "3 | Data Cleaning",
            "4 | Data Transformation",
            "5 | EDA",
        ]
        for i, name in enumerate(names, start=1):
            assert _phase_number(name) == i


class TestRunPipelineOrchestration:

    def _run_with_phases(self, phases, start_from=1):
        with patch("src.pipeline.master_pipeline._PHASES", phases):
            from src.pipeline.master_pipeline import run_pipeline

            try:
                run_pipeline(start_from=start_from)
            except SystemExit:
                pass

    def test_all_phases_called_when_start_from_1(self):
        fns = [MagicMock() for _ in range(3)]
        phases = [_make_phase(i + 1, fns[i]) for i in range(3)]
        self._run_with_phases(phases, start_from=1)
        for fn in fns:
            fn.assert_called_once()

    def test_each_phase_called_with_its_kwargs(self):
        fn = MagicMock()
        phases = [{"name": "1 | Phase 1", "fn": fn, "kwargs": {"x": 99}}]
        self._run_with_phases(phases)
        fn.assert_called_once_with(x=99)

    def test_start_from_skips_earlier_phases(self):
        fns = [MagicMock() for _ in range(4)]
        phases = [_make_phase(i + 1, fns[i]) for i in range(4)]
        self._run_with_phases(phases, start_from=3)
        fns[0].assert_not_called()
        fns[1].assert_not_called()
        fns[2].assert_called_once()
        fns[3].assert_called_once()

    def test_start_from_last_phase_only_runs_last(self):
        fns = [MagicMock() for _ in range(3)]
        phases = [_make_phase(i + 1, fns[i]) for i in range(3)]
        self._run_with_phases(phases, start_from=3)
        fns[0].assert_not_called()
        fns[1].assert_not_called()
        fns[2].assert_called_once()

    def test_phases_run_in_order(self):
        call_order = []

        def make_fn(n):
            def fn(**kwargs):
                call_order.append(n)

            return fn

        phases = [_make_phase(i + 1, make_fn(i + 1)) for i in range(3)]
        self._run_with_phases(phases)
        assert call_order == [1, 2, 3]


class TestRunPipelineFailureHandling:

    def test_failing_phase_stops_pipeline(self):
        fn1 = MagicMock()
        fn2 = MagicMock(side_effect=RuntimeError("boom"))
        fn3 = MagicMock()

        phases = [
            _make_phase(1, fn1),
            _make_phase(2, fn2),
            _make_phase(3, fn3),
        ]

        with patch("src.pipeline.master_pipeline._PHASES", phases):
            from src.pipeline.master_pipeline import run_pipeline

            with pytest.raises(SystemExit):
                run_pipeline(start_from=1)

        fn1.assert_called_once()
        fn2.assert_called_once()
        fn3.assert_not_called()  # must NOT run after failure

    def test_sys_exit_called_on_failure(self):
        fn = MagicMock(side_effect=ValueError("bad"))
        phases = [_make_phase(1, fn)]

        with patch("src.pipeline.master_pipeline._PHASES", phases):
            from src.pipeline.master_pipeline import run_pipeline

            with pytest.raises(SystemExit) as exc_info:
                run_pipeline(start_from=1)
        assert exc_info.value.code == 1

    def test_no_exit_on_full_success(self):
        phases = [_make_phase(i + 1) for i in range(3)]
        with patch("src.pipeline.master_pipeline._PHASES", phases):
            from src.pipeline.master_pipeline import run_pipeline

            # Must complete without raising SystemExit
            run_pipeline(start_from=1)

    def test_invalid_start_from_exits(self):
        phases = [_make_phase(1)]
        with patch("src.pipeline.master_pipeline._PHASES", phases):
            from src.pipeline.master_pipeline import run_pipeline

            with pytest.raises(SystemExit) as exc_info:
                run_pipeline(start_from=99)
        assert exc_info.value.code == 1

    def test_first_phase_failure_does_not_run_second(self):
        fn1 = MagicMock(side_effect=IOError("disk full"))
        fn2 = MagicMock()
        phases = [_make_phase(1, fn1), _make_phase(2, fn2)]

        with patch("src.pipeline.master_pipeline._PHASES", phases):
            from src.pipeline.master_pipeline import run_pipeline

            with pytest.raises(SystemExit):
                run_pipeline()

        fn2.assert_not_called()


class TestPipelineWithRealPhaseNames:
    @pytest.fixture()
    def all_phases_mocked(self):
        import src.pipeline.master_pipeline as mp

        originals = [(i, p["fn"]) for i, p in enumerate(mp._PHASES)]
        mocks = {}
        for i, phase in enumerate(mp._PHASES):
            m = MagicMock(return_value=None)
            mocks[phase["name"]] = m
            mp._PHASES[i] = {**phase, "fn": m}

        yield mocks

        # Restore originals
        for i, orig_fn in originals:
            mp._PHASES[i]["fn"] = orig_fn

    def test_all_real_phases_are_called(self, all_phases_mocked):
        import src.pipeline.master_pipeline as mp

        mp.run_pipeline(start_from=1)
        for name, mock in all_phases_mocked.items():
            assert mock.call_count == 1, f"Phase '{name}' was not called"

    def test_phase_3_skipped_when_starting_from_4(self, all_phases_mocked):
        import src.pipeline.master_pipeline as mp

        mp.run_pipeline(start_from=4)
        for name, mock in all_phases_mocked.items():
            phase_num = mp._phase_number(name)
            if phase_num < 4:
                assert mock.call_count == 0, f"Phase '{name}' should have been skipped"
            else:
                assert mock.call_count == 1, f"Phase '{name}' should have run"

    def test_pipeline_stops_at_phase_2_if_it_fails(self, all_phases_mocked):
        import src.pipeline.master_pipeline as mp

        # Make phase 2 fail
        phase2_name = next(n for n in all_phases_mocked if mp._phase_number(n) == 2)
        all_phases_mocked[phase2_name].side_effect = RuntimeError("validation failed")

        with pytest.raises(SystemExit):
            mp.run_pipeline(start_from=1)

        # Phases after 2 must not have run
        for name, mock in all_phases_mocked.items():
            if mp._phase_number(name) > 2:
                assert (
                    mock.call_count == 0
                ), f"Phase '{name}' should not have run after failure"

    def test_real_phase_kwargs_are_passed(self, all_phases_mocked):
        import src.pipeline.master_pipeline as mp

        mp.run_pipeline(start_from=1)
        for i, phase in enumerate(mp._PHASES):
            mock = all_phases_mocked[phase["name"]]
            _, actual_kwargs = mock.call_args
            assert actual_kwargs == phase["kwargs"], (
                f"Phase '{phase['name']}' received wrong kwargs.\n"
                f"  Expected: {phase['kwargs']}\n"
                f"  Got:      {actual_kwargs}"
            )
