"""Unit tests for src/utils/cluster_diagnostics.py.

All tests use only stdlib — no SLURM, no GPU, no TabPFN required.
Safe to run on any machine (Windows, Linux, macOS, CI).

Run:
    conda activate pfns4neurostim
    pytest tests/test_cluster_diagnostics.py -v
"""
from __future__ import annotations

import io
import os
import sys
import time
from unittest.mock import patch

import pytest

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC_DIR = os.path.join(_PROJECT_ROOT, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from utils.cluster_diagnostics import (
    ClusterDiagnostics,
    _DiagMetrics,
    _GpuPoller,
    _WARNING_RULES,
    _parse_slurm_timelimit,
    _parse_slurm_mem_mb,
    _detect_cluster,
    _wrap_lines,
)


# ---------------------------------------------------------------------------
# _parse_slurm_timelimit
# ---------------------------------------------------------------------------

class TestParseTimelimit:
    def test_hms_format(self):
        assert _parse_slurm_timelimit("4:00:00") == pytest.approx(14400.0)

    def test_hms_format_minutes(self):
        assert _parse_slurm_timelimit("1:30:00") == pytest.approx(5400.0)

    def test_days_hms_format(self):
        # "1-02:30:00" = 1 day + 2h30m = 95400 s
        assert _parse_slurm_timelimit("1-02:30:00") == pytest.approx(95400.0)

    def test_integer_minutes(self):
        assert _parse_slurm_timelimit("240") == pytest.approx(14400.0)

    def test_none_input(self):
        assert _parse_slurm_timelimit(None) is None

    def test_empty_string(self):
        assert _parse_slurm_timelimit("") is None

    def test_unparseable_string(self):
        assert _parse_slurm_timelimit("abc") is None

    def test_mm_ss_format(self):
        # "90:00" = 90 min = 5400 s
        assert _parse_slurm_timelimit("90:00") == pytest.approx(5400.0)


# ---------------------------------------------------------------------------
# _parse_slurm_mem_mb
# ---------------------------------------------------------------------------

class TestParseMemMb:
    def test_integer_mb(self):
        with patch.dict(os.environ, {"SLURM_MEM_PER_NODE": "7168"}):
            assert _parse_slurm_mem_mb() == 7168

    def test_gigabyte_suffix(self):
        with patch.dict(os.environ, {"SLURM_MEM_PER_NODE": "7G"}):
            assert _parse_slurm_mem_mb() == 7 * 1024

    def test_megabyte_suffix(self):
        with patch.dict(os.environ, {"SLURM_MEM_PER_NODE": "7168M"}):
            assert _parse_slurm_mem_mb() == 7168

    def test_absent_env_var(self):
        env = {k: v for k, v in os.environ.items() if k != "SLURM_MEM_PER_NODE"}
        with patch.dict(os.environ, env, clear=True):
            assert _parse_slurm_mem_mb() is None

    def test_empty_string(self):
        with patch.dict(os.environ, {"SLURM_MEM_PER_NODE": ""}):
            assert _parse_slurm_mem_mb() is None

    def test_lowercase_g_suffix(self):
        with patch.dict(os.environ, {"SLURM_MEM_PER_NODE": "32g"}):
            assert _parse_slurm_mem_mb() == 32 * 1024


# ---------------------------------------------------------------------------
# _wrap_lines
# ---------------------------------------------------------------------------

class TestWrapLines:
    def test_short_string_unchanged(self):
        assert _wrap_lines("hello", 40) == ["hello"]

    def test_long_string_wrapped(self):
        text = "a " * 30  # 60 chars
        lines = _wrap_lines(text, 20)
        assert all(len(l) <= 20 for l in lines)

    def test_preserves_newlines(self):
        text = "line one\nline two"
        result = _wrap_lines(text, 80)
        assert result == ["line one", "line two"]

    def test_empty_paragraph_yields_empty_string(self):
        result = _wrap_lines("first\n\nsecond", 80)
        assert '' in result


# ---------------------------------------------------------------------------
# _detect_cluster
# ---------------------------------------------------------------------------

class TestDetectCluster:
    def test_slurm_cluster_name_mila(self):
        with patch.dict(os.environ, {"SLURM_CLUSTER_NAME": "mila"}):
            assert _detect_cluster() == "mila"

    def test_slurm_cluster_name_cedar(self):
        with patch.dict(os.environ, {"SLURM_CLUSTER_NAME": "cedar"}):
            assert _detect_cluster() == "cedar"

    def test_fallback_unknown(self):
        env = {k: v for k, v in os.environ.items() if k != "SLURM_CLUSTER_NAME"}
        with patch.dict(os.environ, env, clear=True):
            # Hostname won't match any known cluster on a dev machine
            result = _detect_cluster()
            assert isinstance(result, str)


# ---------------------------------------------------------------------------
# ClusterDiagnostics.record_experiment
# ---------------------------------------------------------------------------

class TestRecordExperiment:
    def test_increments_counter_when_enabled(self):
        diag = ClusterDiagnostics(enabled=True)
        diag._t0 = time.time()
        diag.record_experiment(n_completed=3)
        assert diag._metrics.n_experiments_completed == 3

    def test_multiple_increments(self):
        diag = ClusterDiagnostics(enabled=True)
        diag._t0 = time.time()
        for _ in range(5):
            diag.record_experiment(n_completed=1)
        assert diag._metrics.n_experiments_completed == 5

    def test_no_op_when_disabled(self):
        diag = ClusterDiagnostics(enabled=False)
        diag.record_experiment(n_completed=99)
        # Counter should stay at 0 (no-op)
        assert diag._metrics.n_experiments_completed == 0


# ---------------------------------------------------------------------------
# ClusterDiagnostics — no-op when disabled
# ---------------------------------------------------------------------------

class TestNopWhenDisabled:
    def test_exit_does_not_print(self, capsys):
        with ClusterDiagnostics(enabled=False):
            pass
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_enter_returns_self(self):
        diag = ClusterDiagnostics(enabled=False)
        result = diag.__enter__()
        assert result is diag
        diag.__exit__(None, None, None)

    def test_does_not_suppress_exceptions(self):
        with pytest.raises(ValueError):
            with ClusterDiagnostics(enabled=False):
                raise ValueError("test exception")


# ---------------------------------------------------------------------------
# _compute_grade — known input → known output
# ---------------------------------------------------------------------------

class TestComputeGrade:
    def _make_diag_with_metrics(self, **overrides) -> ClusterDiagnostics:
        diag = ClusterDiagnostics(enabled=True)
        diag._t0 = time.time()
        for k, v in overrides.items():
            setattr(diag._metrics, k, v)
        return diag

    def test_no_data_returns_question_mark(self):
        diag = self._make_diag_with_metrics()
        assert diag._compute_grade() == '?'

    def test_high_efficiency_grades_A(self):
        # GPU mem 99%, walltime 99%, GPU util 99% → score ~99 → A
        GB = 1024 ** 3
        diag = self._make_diag_with_metrics(
            cuda_available=True,
            peak_gpu_mem_bytes=int(6.93 * GB),
            requested_mem_bytes=int(7.0 * GB),
            elapsed_s=3.96 * 3600,
            slurm_timelimit_s=4.0 * 3600,
            gpu_util_samples=[99, 98, 99],
            nvidia_smi_available=True,
        )
        assert diag._compute_grade() == 'A'

    def test_low_efficiency_grades_F(self):
        GB = 1024 ** 3
        diag = self._make_diag_with_metrics(
            cuda_available=True,
            peak_gpu_mem_bytes=int(0.5 * GB),
            requested_mem_bytes=int(7.0 * GB),    # only 7% used
            elapsed_s=0.3 * 3600,
            slurm_timelimit_s=4.0 * 3600,          # only 7.5% used
            gpu_util_samples=[5, 3, 4],            # almost idle
            nvidia_smi_available=True,
        )
        assert diag._compute_grade() == 'F'

    def test_walltime_only_data(self):
        # Only walltime data → grade based on walltime only
        diag = self._make_diag_with_metrics(
            elapsed_s=3.6 * 3600,
            slurm_timelimit_s=4.0 * 3600,   # 90% → should be A
        )
        assert diag._compute_grade() == 'A'


# ---------------------------------------------------------------------------
# _generate_warnings — rules fire correctly
# ---------------------------------------------------------------------------

class TestGenerateWarnings:
    GB = 1024 ** 3

    def _make_diag(self, **overrides) -> ClusterDiagnostics:
        diag = ClusterDiagnostics(enabled=True)
        diag._t0 = time.time()
        for k, v in overrides.items():
            setattr(diag._metrics, k, v)
        return diag

    def test_gpu_mem_underuse_fires(self):
        diag = self._make_diag(
            cuda_available=True,
            peak_gpu_mem_bytes=int(3.0 * self.GB),
            requested_mem_bytes=int(7.0 * self.GB),  # 43% → fires
        )
        ids = [w['id'] for w in diag._generate_warnings()]
        assert 'GPU_MEM_UNDERUSE' in ids

    def test_gpu_mem_underuse_does_not_fire_when_efficient(self):
        diag = self._make_diag(
            cuda_available=True,
            peak_gpu_mem_bytes=int(6.0 * self.GB),
            requested_mem_bytes=int(7.0 * self.GB),  # 86% → no fire
        )
        ids = [w['id'] for w in diag._generate_warnings()]
        assert 'GPU_MEM_UNDERUSE' not in ids

    def test_walltime_overrequest_fires(self):
        diag = self._make_diag(
            elapsed_s=0.4 * 3600,
            slurm_timelimit_s=4.0 * 3600,   # 10% → fires
        )
        ids = [w['id'] for w in diag._generate_warnings()]
        assert 'WALLTIME_OVERREQUEST' in ids

    def test_walltime_overrequest_does_not_fire_when_close(self):
        diag = self._make_diag(
            elapsed_s=3.0 * 3600,
            slurm_timelimit_s=4.0 * 3600,   # 75% → no fire
        )
        ids = [w['id'] for w in diag._generate_warnings()]
        assert 'WALLTIME_OVERREQUEST' not in ids

    def test_cuda_fragmentation_fires(self):
        diag = self._make_diag(
            cuda_available=True,
            peak_gpu_mem_bytes=int(3.0 * self.GB),
            reserved_gpu_mem_bytes=int(5.0 * self.GB),  # 40% frag → fires
        )
        ids = [w['id'] for w in diag._generate_warnings()]
        assert 'CUDA_FRAGMENTATION' in ids

    def test_cpu_underuse_fires_when_gt_2(self):
        diag = self._make_diag(n_cpus_requested=4)
        ids = [w['id'] for w in diag._generate_warnings()]
        assert 'CPU_UNDERUSE' in ids

    def test_cpu_underuse_does_not_fire_for_default_2(self):
        diag = self._make_diag(n_cpus_requested=2)
        ids = [w['id'] for w in diag._generate_warnings()]
        assert 'CPU_UNDERUSE' not in ids

    def test_ram_underuse_fires(self):
        diag = self._make_diag(
            peak_rss_bytes=int(2.0 * self.GB),
            requested_ram_bytes=int(7.0 * self.GB),  # 29% → fires
        )
        ids = [w['id'] for w in diag._generate_warnings()]
        assert 'RAM_UNDERUSE' in ids

    def test_no_warnings_when_all_efficient(self):
        diag = self._make_diag(
            cuda_available=True,
            peak_gpu_mem_bytes=int(6.5 * self.GB),
            requested_mem_bytes=int(7.0 * self.GB),
            reserved_gpu_mem_bytes=int(6.6 * self.GB),
            elapsed_s=3.5 * 3600,
            slurm_timelimit_s=4.0 * 3600,
            gpu_util_samples=[80, 82, 79],
            nvidia_smi_available=True,
            peak_rss_bytes=int(5.5 * self.GB),
            requested_ram_bytes=int(7.0 * self.GB),
            n_cpus_requested=2,
        )
        warnings = diag._generate_warnings()
        assert warnings == [], f"Expected no warnings, got: {[w['id'] for w in warnings]}"


# ---------------------------------------------------------------------------
# format_terminal_report — structural checks
# ---------------------------------------------------------------------------

class TestFormatTerminalReport:
    BOX_WIDTH = 72

    def _make_diag(self, **overrides) -> ClusterDiagnostics:
        diag = ClusterDiagnostics(enabled=True)
        diag._t0 = time.time()
        for k, v in overrides.items():
            setattr(diag._metrics, k, v)
        return diag

    def test_all_lines_within_box_width(self):
        diag = self._make_diag(
            experiment_tag='nhp-test-abc12',
            elapsed_s=120.0,
        )
        report = diag.format_terminal_report()
        for i, line in enumerate(report.splitlines()):
            assert len(line) <= self.BOX_WIDTH, (
                f"Line {i} exceeds {self.BOX_WIDTH} chars: {len(line)!r}\n{line!r}"
            )

    def test_header_present(self):
        diag = self._make_diag(experiment_tag='nhp-test-abc12')
        report = diag.format_terminal_report()
        assert 'CLUSTER DIAGNOSTICS' in report

    def test_grade_present(self):
        diag = self._make_diag()
        report = diag.format_terminal_report()
        assert 'EFFICIENCY GRADE' in report

    def test_no_warnings_message_when_clean(self):
        diag = self._make_diag()
        report = diag.format_terminal_report()
        assert 'No warnings' in report

    def test_warnings_section_appears(self):
        GB = 1024 ** 3
        diag = self._make_diag(
            elapsed_s=0.2 * 3600,
            slurm_timelimit_s=4.0 * 3600,  # fires WALLTIME_OVERREQUEST
        )
        report = diag.format_terminal_report()
        assert 'WARNINGS' in report
        assert 'WALLTIME' in report or 'Walltime' in report

    def test_report_is_string(self):
        diag = self._make_diag()
        assert isinstance(diag.format_terminal_report(), str)

    def test_context_manager_prints_report(self, capsys):
        with ClusterDiagnostics(tag='test', device='cpu', n_planned=0, enabled=True):
            pass
        captured = capsys.readouterr()
        assert 'CLUSTER DIAGNOSTICS' in captured.out

    def test_tag_appears_in_report(self):
        diag = self._make_diag(experiment_tag='my-unique-experiment-tag')
        report = diag.format_terminal_report()
        assert 'my-unique-experiment-tag' in report


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_zero_elapsed_no_crash(self):
        diag = ClusterDiagnostics(enabled=True)
        diag._metrics.elapsed_s = 0.0
        # Should not raise
        _ = diag.format_terminal_report()

    def test_exception_inside_ctx_not_suppressed(self):
        with pytest.raises(RuntimeError):
            with ClusterDiagnostics(enabled=True):
                raise RuntimeError("experiment failed")

    def test_exception_inside_disabled_ctx_not_suppressed(self):
        with pytest.raises(RuntimeError):
            with ClusterDiagnostics(enabled=False):
                raise RuntimeError("experiment failed")

    def test_all_slurm_vars_absent_no_crash(self):
        env_clean = {
            k: v for k, v in os.environ.items()
            if not k.startswith('SLURM_')
        }
        with patch.dict(os.environ, env_clean, clear=True):
            with ClusterDiagnostics(tag='no-slurm', enabled=True):
                pass   # should print gracefully with ?? rows

    def test_n_planned_zero_no_division_error(self):
        diag = ClusterDiagnostics(tag='t', n_planned=0, enabled=True)
        diag._t0 = time.time()
        diag._metrics.elapsed_s = 60.0
        _ = diag.format_terminal_report()
