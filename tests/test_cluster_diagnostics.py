"""Unit tests for src/pfns4neurostim/diagnostics/cluster.py.

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

from pfns4neurostim.diagnostics.cluster import (
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

    def test_gpu_mem_headroom_fires_against_the_card_not_the_mem_request(self):
        """Renamed and re-based on 2026-10-01: --gres=gpu:1 asks for a whole card and --mem is HOST RAM,
        so GPU memory has to be judged against the device's VRAM. Comparing it with --mem graded every
        lane-parallel job F while its GPU utilisation was 59-67%."""
        diag = self._make_diag(
            cuda_available=True,
            peak_gpu_mem_bytes=int(1.0 * self.GB),
            total_gpu_mem_bytes=int(40.0 * self.GB),   # 1 of 40 GB = 2.5% -> fires
        )
        ids = [w['id'] for w in diag._generate_warnings()]
        assert 'GPU_MEM_HEADROOM' in ids

    def test_gpu_mem_headroom_counts_every_lane(self):
        """Lanes share one card, so the job's footprint is lanes x the per-process peak."""
        diag = self._make_diag(
            cuda_available=True,
            n_lanes=8,
            peak_gpu_mem_bytes=int(2.0 * self.GB),
            total_gpu_mem_bytes=int(40.0 * self.GB),   # 8 x 2 = 16 of 40 GB = 40% -> no fire
        )
        ids = [w['id'] for w in diag._generate_warnings()]
        assert 'GPU_MEM_HEADROOM' not in ids

    def test_gpu_mem_is_silent_without_the_card_capacity(self):
        """No VRAM figure means no denominator; it must not fall back to --mem."""
        diag = self._make_diag(
            cuda_available=True,
            peak_gpu_mem_bytes=int(1.0 * self.GB),
            requested_mem_bytes=int(40.0 * self.GB),
        )
        ids = [w['id'] for w in diag._generate_warnings()]
        assert 'GPU_MEM_HEADROOM' not in ids

    # --- cluster-shaped recipes and the Narval MIG-slice rule (2026-10-02) ---------------------------------

    def _fired(self, diag, rule_id):
        return {w['id']: w for w in diag._generate_warnings()}.get(rule_id)

    def test_slice_oversized_fires_on_a_whole_a100_on_narval(self):
        """Job 4487625 peaked at 0.08 GB; a 5 GB slice holds it, so the whole card is the wrong request."""
        diag = self._make_diag(
            cluster_name='narval', cuda_available=True,
            peak_gpu_mem_bytes=int(0.5 * self.GB), total_gpu_mem_bytes=int(40.0 * self.GB),
        )
        w = self._fired(diag, 'GPU_SLICE_OVERSIZED')
        assert w is not None
        assert 'a100_1g.5gb' in w['rendered_text'] and '--gpu-type a100_1g.5gb' in w['rendered_fix']

    def test_slice_rule_picks_the_smallest_slice_with_headroom(self):
        diag = self._make_diag(
            cluster_name='narval', cuda_available=True,
            peak_gpu_mem_bytes=int(5.0 * self.GB), total_gpu_mem_bytes=int(40.0 * self.GB),
        )   # 5 GB x 1.25 = 6.25 GB: the 5 GB slice is too small, the 10 GB one fits
        assert 'a100_2g.10gb' in self._fired(diag, 'GPU_SLICE_OVERSIZED')['rendered_text']

    def test_slice_rule_is_silent_off_narval_inside_a_slice_and_when_the_job_needs_the_card(self):
        base = dict(cuda_available=True, peak_gpu_mem_bytes=int(0.5 * self.GB))
        mila = self._make_diag(cluster_name='mila', total_gpu_mem_bytes=int(40.0 * self.GB), **base)
        in_slice = self._make_diag(cluster_name='narval', total_gpu_mem_bytes=int(5.0 * self.GB), **base)
        needs_card = self._make_diag(
            cluster_name='narval', cuda_available=True,
            peak_gpu_mem_bytes=int(18.0 * self.GB), total_gpu_mem_bytes=int(40.0 * self.GB),
        )
        for diag in (mila, in_slice, needs_card):
            assert self._fired(diag, 'GPU_SLICE_OVERSIZED') is None

    def test_recipes_use_narval_options_there_and_sbatch_lines_elsewhere(self):
        kw = dict(
            requested_ram_bytes=int(10.0 * self.GB), peak_rss_bytes=int(1.0 * self.GB), n_lanes=1,
        )
        narval = self._fired(self._make_diag(cluster_name='narval', **kw), 'RAM_UNDERUSE')['rendered_fix']
        mila = self._fired(self._make_diag(cluster_name='mila', **kw), 'RAM_UNDERUSE')['rendered_fix']
        assert 'narval.sh do sbatch ... --mem 2G' in narval and '#SBATCH' not in narval
        assert '#SBATCH --mem=2G' in mila

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


# ---------------------------------------------------------------------------
# Time limit fallback and template text (2026-09-20 fixes)
# ---------------------------------------------------------------------------
class TestTimelimitAndText:
    def test_env_var_wins(self) -> None:
        from pfns4neurostim.diagnostics.cluster import _read_slurm_timelimit

        with patch.dict(os.environ, {"SLURM_TIMELIMIT": "1:00:00"}):
            assert _read_slurm_timelimit() == 3600.0

    def test_falls_back_to_squeue(self) -> None:
        from types import SimpleNamespace

        from pfns4neurostim.diagnostics.cluster import _read_slurm_timelimit

        env = {k: v for k, v in os.environ.items() if k != "SLURM_TIMELIMIT"}
        env["SLURM_JOB_ID"] = "123"
        with patch.dict(os.environ, env, clear=True), patch(
            "pfns4neurostim.diagnostics.cluster.subprocess.run",
            return_value=SimpleNamespace(stdout="1-00:00:00\n"),
        ) as run:
            assert _read_slurm_timelimit() == 86400.0
        assert run.call_args[0][0][:4] == ["squeue", "-h", "-j", "123"]

    def test_no_job_no_limit(self) -> None:
        from pfns4neurostim.diagnostics.cluster import _read_slurm_timelimit

        env = {k: v for k, v in os.environ.items() if k not in ("SLURM_TIMELIMIT", "SLURM_JOB_ID")}
        with patch.dict(os.environ, env, clear=True):
            assert _read_slurm_timelimit() is None

    def test_warning_templates_have_no_doubled_percent(self) -> None:
        import inspect

        from pfns4neurostim.diagnostics import cluster

        assert "%%" not in inspect.getsource(cluster)


class TestLaneAwareness:
    """Lanes multiply every per-process measurement (the rubric's project-specific joint rule).

    Added 2026-10-01 after the report advised `--mem=1G` for jobs whose job-level MaxRSS was measured at
    3.75 GB: it had compared one lane's 1.03 GB against the job's 10 GB allocation.
    """

    GB = 1024 ** 3

    def _diag(self, **kwargs):
        from pfns4neurostim.diagnostics.cluster import ClusterDiagnostics
        diag = ClusterDiagnostics(enabled=True, n_lanes=kwargs.pop('n_lanes', 1))
        for k, v in kwargs.items():
            setattr(diag._metrics, k, v)
        diag._metrics.n_lanes = diag._n_lanes
        return diag

    def test_job_peak_rss_multiplies_by_lanes(self):
        diag = self._diag(n_lanes=4, peak_rss_bytes=int(1.03 * self.GB))
        assert diag._metrics.job_peak_rss_bytes == int(1.03 * self.GB) * 4

    def test_ram_underuse_judges_the_job_not_the_lane(self):
        """4 x 1.03 = 4.12 GB of 10 GB is 41%: still under the 70% band, but the SUGGESTION must cover
        the job. The old rule's `--mem=1G` would have OOM-killed it."""
        diag = self._diag(n_lanes=4, peak_rss_bytes=int(1.03 * self.GB),
                          requested_ram_bytes=10 * self.GB)
        fired = {w['id']: w for w in diag._generate_warnings()}
        assert 'RAM_UNDERUSE' in fired
        assert '4.1 GB peak' in fired['RAM_UNDERUSE']['rendered_text']
        # 4.12 GB x 1.25 = 5.15 GB: rounded UP (2026-10-02) the request is 6G; to-nearest gave 5G, below peak + 25%.
        assert '--mem=6G' in fired['RAM_UNDERUSE']['rendered_fix']

    def test_ram_underuse_silent_when_the_lanes_fill_the_allocation(self):
        diag = self._diag(n_lanes=8, peak_rss_bytes=int(1.05 * self.GB),
                          requested_ram_bytes=10 * self.GB)   # 8.4 of 10 GB = 84%
        assert 'RAM_UNDERUSE' not in [w['id'] for w in diag._generate_warnings()]

    def test_cpu_underuse_silent_when_every_core_has_a_lane(self):
        """Measured on job 11012813: sstat AveCPU 08:42:54 vs CPUTime 08:43:20 = 99.9% of 4 cores,
        with LANES=4. The old rule fired anyway and advised halving the cores."""
        diag = self._diag(n_lanes=4, n_cpus_requested=4)
        assert diag._metrics.idle_cpus == 0
        assert 'CPU_UNDERUSE' not in [w['id'] for w in diag._generate_warnings()]

    def test_cpu_underuse_fires_on_genuinely_idle_cores(self):
        diag = self._diag(n_lanes=2, n_cpus_requested=8)
        fired = {w['id']: w for w in diag._generate_warnings()}
        assert fired['CPU_UNDERUSE']['rendered_text'].count('6 core(s)') == 1
        assert '--cpus-per-task=2' in fired['CPU_UNDERUSE']['rendered_fix']

    def test_cpu_underuse_respects_the_two_core_baseline(self):
        """One lane on two cores is the accepted minimum, not something to warn about."""
        diag = self._diag(n_lanes=1, n_cpus_requested=2)
        assert 'CPU_UNDERUSE' not in [w['id'] for w in diag._generate_warnings()]


class TestPerRulePercentages:
    """Each warning's {pct} is its own metric's share.

    Before 2026-10-01 a single `pct` key was overwritten by whichever block ran last, so the GPU and
    walltime warnings printed the RAM percentage: job 11012807 reported "used 0.65h of 12.00h (10%)"
    where the true share is 5%.
    """

    GB = 1024 ** 3

    def _diag(self, **kwargs):
        from pfns4neurostim.diagnostics.cluster import ClusterDiagnostics
        diag = ClusterDiagnostics(enabled=True, n_lanes=kwargs.pop('n_lanes', 1))
        for k, v in kwargs.items():
            setattr(diag._metrics, k, v)
        diag._metrics.n_lanes = diag._n_lanes
        return diag

    def test_walltime_pct_is_the_time_share(self):
        diag = self._diag(
            n_lanes=4, elapsed_s=0.65 * 3600, slurm_timelimit_s=12 * 3600,
            peak_rss_bytes=int(1.03 * self.GB), requested_ram_bytes=10 * self.GB,
        )
        fired = {w['id']: w for w in diag._generate_warnings()}
        text = fired['WALLTIME_OVERREQUEST']['rendered_text']
        assert '(5%)' in text, text          # 0.65 / 12, NOT the 41% RAM figure
        assert 'Fairshare' not in text       # SLURM charges elapsed, not requested

    def test_ram_pct_is_the_memory_share(self):
        diag = self._diag(
            n_lanes=4, elapsed_s=0.65 * 3600, slurm_timelimit_s=12 * 3600,
            peak_rss_bytes=int(1.03 * self.GB), requested_ram_bytes=10 * self.GB,
        )
        fired = {w['id']: w for w in diag._generate_warnings()}
        assert '(41%)' in fired['RAM_UNDERUSE']['rendered_text']
