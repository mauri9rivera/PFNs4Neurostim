"""Cluster diagnostics context manager for SLURM HPC jobs.

Produces a terminal summary block (printed to the SLURM .out log at job end)
covering GPU memory, GPU utilisation, CUDA fragmentation, walltime efficiency,
CPU/RAM efficiency, experiments-per-GPU-hour throughput, an A–F efficiency
grade, and named best-practice warnings with corrected #SBATCH fix recipes.

Designed for stdlib-only operation (Python 3.9+, PyTorch): time, threading,
subprocess, resource (lazy), os, socket, textwrap.  No psutil, gputil, or
py3nvml required.

Gracefully degrades to a no-op when:
  - ``enabled=False`` (flag not passed)
  - SLURM env vars absent (local laptop run)
  - CUDA not available
  - nvidia-smi not found (CPU-only node or container)
  - resource module absent (Windows)

Works identically on Mila (RTX8000) and Alliance Canada / CC clusters
(V100/A100, Cedar/Narval/Beluga/Graham).

Usage::

    with ClusterDiagnostics(
        tag=save_tag, device=device, n_planned=len(experiments),
        enabled=args.cluster_diag,
    ) as diag:
        for subj_idx, emg_idx in experiments:
            result = evaluate_optimization(...)
            diag.record_experiment(n_completed=1)
"""
from __future__ import annotations

import os
import subprocess
import textwrap
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class _DiagMetrics:
    """Raw collected metrics.  Fields are None / 0 until populated."""

    # Timing
    elapsed_s: float = 0.0
    slurm_timelimit_s: Optional[float] = None

    # GPU memory (bytes)
    peak_gpu_mem_bytes: int = 0
    reserved_gpu_mem_bytes: int = 0
    requested_mem_bytes: Optional[int] = None       # from SLURM_MEM_PER_NODE

    # GPU utilisation (integer 0–100 per sample)
    gpu_util_samples: List[int] = field(default_factory=list)

    # CPU / RAM
    peak_rss_bytes: int = 0
    requested_ram_bytes: Optional[int] = None       # same as requested_mem_bytes
    n_cpus_requested: Optional[int] = None          # SLURM_CPUS_PER_TASK

    # Throughput
    n_experiments_completed: int = 0
    n_experiments_planned: int = 0

    # Context
    cluster_name: str = 'unknown'
    job_id: Optional[str] = None
    array_task_id: Optional[str] = None
    experiment_tag: str = ''
    cuda_available: bool = False
    nvidia_smi_available: bool = False


# ---------------------------------------------------------------------------
# Background GPU poller
# ---------------------------------------------------------------------------

class _GpuPoller(threading.Thread):
    """Daemon thread that polls ``nvidia-smi`` every *interval_s* seconds.

    Args:
        interval_s: Polling interval in seconds (default 30).

    Attributes:
        samples: List of integer GPU utilisation percentages (0–100).
            If multiple GPUs are visible, the mean across all GPUs is stored.
        available: ``True`` if the first probe returned valid output.
    """

    def __init__(self, interval_s: int = 30) -> None:
        super().__init__(daemon=True)
        self.interval_s = interval_s
        self.samples: List[int] = []
        self.available: bool = False
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------
    def run(self) -> None:
        """Polling loop — runs until :meth:`stop` is called."""
        result = self._query()
        if result is None:
            return
        self.available = True
        self.samples.append(result)

        while not self._stop_event.wait(self.interval_s):
            val = self._query()
            if val is not None:
                self.samples.append(val)

    def stop(self) -> None:
        """Signal the polling loop to terminate."""
        self._stop_event.set()

    def _query(self) -> Optional[int]:
        """Run ``nvidia-smi`` and return mean GPU utilisation across visible GPUs.

        Returns:
            Integer utilisation 0–100, or ``None`` on any failure.
        """
        try:
            out = subprocess.run(
                ['nvidia-smi',
                 '--query-gpu=utilization.gpu',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=10,
            )
            if out.returncode != 0:
                return None
            lines = [ln.strip() for ln in out.stdout.strip().splitlines() if ln.strip()]
            vals: List[int] = []
            for ln in lines:
                try:
                    vals.append(int(ln))
                except ValueError:
                    pass
            return int(sum(vals) / len(vals)) if vals else None
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError, ValueError):
            return None


# ---------------------------------------------------------------------------
# Env-var parsers
# ---------------------------------------------------------------------------

def _detect_cluster() -> str:
    """Detect cluster from SLURM env vars or hostname.

    Checks ``SLURM_CLUSTER_NAME`` first (most reliable).  Falls back to a
    hostname substring match for Cedar / Narval / Beluga / Graham / Mila.

    Returns:
        Lowercase cluster identifier: ``'mila'``, ``'cedar'``, ``'narval'``,
        ``'beluga'``, ``'graham'``, ``'niagara'``, or ``'unknown'``.
    """
    name = os.environ.get('SLURM_CLUSTER_NAME', '').lower().strip()
    if name:
        return name
    try:
        import socket
        hostname = socket.gethostname().lower()
    except OSError:
        return 'unknown'
    for known in ('mila', 'cedar', 'narval', 'beluga', 'graham', 'niagara'):
        if known in hostname:
            return known
    return 'unknown'


def _parse_slurm_timelimit(raw: Optional[str]) -> Optional[float]:
    """Parse ``SLURM_TIMELIMIT`` to total seconds.

    Supported formats observed in the wild::

        "4:00:00"       → HH:MM:SS
        "1-04:00:00"    → D-HH:MM:SS
        "240"           → integer minutes

    Args:
        raw: Raw string value of ``SLURM_TIMELIMIT``, or ``None``.

    Returns:
        Total seconds as ``float``, or ``None`` if absent or unparseable.
    """
    if not raw or not raw.strip():
        return None
    raw = raw.strip()
    try:
        if raw.isdigit():
            # Pure integer → minutes
            return float(raw) * 60.0
        if '-' in raw:
            day_part, time_part = raw.split('-', 1)
            days = int(day_part)
        else:
            days = 0
            time_part = raw
        parts = time_part.split(':')
        if len(parts) == 3:
            h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
        elif len(parts) == 2:
            h, m, s = 0, int(parts[0]), int(parts[1])
        else:
            return None
        return float(days * 86400 + h * 3600 + m * 60 + s)
    except (ValueError, IndexError):
        return None


def _parse_slurm_mem_mb() -> Optional[int]:
    """Parse ``SLURM_MEM_PER_NODE`` to megabytes.

    ``SLURM_MEM_PER_NODE`` is documented to be an integer in MB, but some
    cluster configurations append a suffix (M/G/T).

    Returns:
        Integer megabytes, or ``None`` if env var absent or unparseable.
    """
    raw = os.environ.get('SLURM_MEM_PER_NODE', '').strip()
    if not raw:
        return None
    suffix = raw[-1].upper() if raw[-1].isalpha() else ''
    raw_num = raw[:-1] if suffix else raw
    try:
        val = int(raw_num)
    except ValueError:
        return None
    if suffix in ('', 'M'):
        return val
    elif suffix == 'G':
        return val * 1024
    elif suffix == 'T':
        return val * 1024 * 1024
    return val


def _collect_resource_metrics() -> Tuple[int, Optional[int]]:
    """Collect peak RSS and requested CPU count from OS and SLURM env.

    Uses the stdlib ``resource`` module (Linux / macOS only).  Returns zeros
    gracefully on Windows or if the module is unavailable.

    Returns:
        ``(peak_rss_bytes, n_cpus_requested)`` where ``n_cpus_requested`` is
        ``None`` if ``SLURM_CPUS_PER_TASK`` is absent.
    """
    peak_rss = 0
    try:
        import resource as _res
        import platform
        usage = _res.getrusage(_res.RUSAGE_SELF)
        if platform.system() == 'Darwin':
            peak_rss = usage.ru_maxrss            # bytes on macOS
        else:
            peak_rss = usage.ru_maxrss * 1024     # KB → bytes on Linux
    except (ImportError, AttributeError, OSError):
        pass

    n_cpus: Optional[int] = None
    raw_cpus = os.environ.get('SLURM_CPUS_PER_TASK', '').strip()
    if raw_cpus.isdigit():
        n_cpus = int(raw_cpus)

    return peak_rss, n_cpus


# ---------------------------------------------------------------------------
# Text helper
# ---------------------------------------------------------------------------

def _wrap_lines(text: str, width: int) -> List[str]:
    """Word-wrap *text* to *width*, preserving explicit newlines.

    Args:
        text: Input string (may contain ``\\n``).
        width: Maximum character width per output line.

    Returns:
        List of wrapped line strings (never empty — blank paragraphs yield
        a single empty-string entry).
    """
    result: List[str] = []
    for para in text.splitlines():
        wrapped = textwrap.wrap(para, width)
        result.extend(wrapped if wrapped else [''])
    return result


# ---------------------------------------------------------------------------
# Warning rules
# ---------------------------------------------------------------------------

# Each rule dict has:
#   'id':        machine-readable name
#   'condition': callable(_DiagMetrics) -> bool — fires when True
#   'text':      warning text template (Python .format(**vars))
#   'fix':       fix recipe template (Python .format(**vars))

_WARNING_RULES: List[Dict[str, Any]] = [
    {
        'id': 'GPU_MEM_UNDERUSE',
        'condition': lambda m: (
            m.cuda_available
            and m.requested_mem_bytes is not None
            and m.requested_mem_bytes > 0
            and m.peak_gpu_mem_bytes > 0
            and (m.peak_gpu_mem_bytes / m.requested_mem_bytes) < 0.75
        ),
        'text': (
            'GPU_MEM_UNDERUSE: Peak GPU memory {peak_gpu_gb:.1f} GB is below '
            '75%% of requested {req_gb:.1f} GB ({pct:.0f}%% utilisation).'
        ),
        'fix': (
            'Reduce --mem to peak + 20%% headroom:\n'
            '  #SBATCH --mem={suggested_mem_gb:.0f}G'
        ),
    },
    {
        'id': 'WALLTIME_OVERREQUEST',
        'condition': lambda m: (
            m.slurm_timelimit_s is not None
            and m.slurm_timelimit_s > 0
            and (m.elapsed_s / m.slurm_timelimit_s) < 0.55
        ),
        'text': (
            'WALLTIME_OVERREQUEST: Job used {elapsed_h:.2f}h of '
            '{limit_h:.2f}h requested ({pct:.0f}%%). '
            'Over-requesting time lowers your Fairshare score.'
        ),
        'fix': (
            'Reduce --time to elapsed + 20%% rounded to 15 min:\n'
            '  #SBATCH --time={suggested_time}'
        ),
    },
    {
        'id': 'GPU_IDLE',
        'condition': lambda m: (
            m.nvidia_smi_available
            and len(m.gpu_util_samples) >= 2
            and (sum(m.gpu_util_samples) / len(m.gpu_util_samples)) < 50
        ),
        'text': (
            'GPU_IDLE: Mean GPU utilisation {mean_util:.0f}%% < 50%%. '
            'The GPU is idle for more than half the job.'
        ),
        'fix': (
            'Profile with torch.profiler to find idle windows.\n'
            'Common causes: data loading between reps, Python GC, or\n'
            'CPU-bound preprocessing. Move load_data() outside the inner loop.'
        ),
    },
    {
        'id': 'CUDA_FRAGMENTATION',
        'condition': lambda m: (
            m.cuda_available
            and m.reserved_gpu_mem_bytes > 0
            and m.peak_gpu_mem_bytes > 0
            and (
                (m.reserved_gpu_mem_bytes - m.peak_gpu_mem_bytes)
                / m.reserved_gpu_mem_bytes
            ) > 0.25
        ),
        'text': (
            'CUDA_FRAGMENTATION: PyTorch reserved {res_gb:.2f} GB but peak '
            'allocation was only {peak_gb:.2f} GB ({frag_pct:.0f}%% wasted by '
            'allocator caching).'
        ),
        'fix': (
            'Call torch.cuda.empty_cache() between subjects/experiments, or set:\n'
            '  export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128'
        ),
    },
    {
        'id': 'CPU_UNDERUSE',
        'condition': lambda m: (
            m.n_cpus_requested is not None
            and m.n_cpus_requested > 2
        ),
        'text': (
            'CPU_UNDERUSE: Job requested {n_cpus} CPUs. '
            'TabPFN BO loops are single-threaded; extra CPUs sit idle.'
        ),
        'fix': (
            'Unless you are using multi-threaded data loading or '
            'numpy parallelism, reduce to 2:\n'
            '  #SBATCH --cpus-per-task=2'
        ),
    },
    {
        'id': 'RAM_UNDERUSE',
        'condition': lambda m: (
            m.requested_ram_bytes is not None
            and m.requested_ram_bytes > 0
            and m.peak_rss_bytes > 0
            and (m.peak_rss_bytes / m.requested_ram_bytes) < 0.70
        ),
        'text': (
            'RAM_UNDERUSE: Peak RSS {peak_rss_gb:.1f} GB is below '
            '70%% of requested RAM {req_ram_gb:.1f} GB ({pct:.0f}%%).'
        ),
        'fix': (
            'Reduce --mem to peak RSS + 25%% headroom:\n'
            '  #SBATCH --mem={suggested_mem_gb:.0f}G'
        ),
    },
]


# ---------------------------------------------------------------------------
# Main context manager
# ---------------------------------------------------------------------------

class ClusterDiagnostics:
    """Context manager that collects HPC job efficiency metrics and prints a
    teaching-signal summary block at job end.

    Designed to wrap the outermost experiment loop in each CLI entry point.
    When ``enabled=False``, all methods are no-ops and ``__exit__`` prints
    nothing — zero overhead on laptop / CI runs.

    Args:
        tag: Experiment tag string used in the report header.
        device: PyTorch device string (``'cpu'`` or ``'cuda'``).
        n_planned: Total number of ``(subject, emg)`` pairs planned for this
            SLURM job.  Used for the throughput metric denominator.
        poll_interval_s: GPU utilisation polling interval in seconds (default
            30 — keeps overhead negligible even for short jobs).
        enabled: ``False`` → pure no-op context manager.

    Example::

        with ClusterDiagnostics(
            tag=save_tag, device=device, n_planned=len(experiments),
            enabled=args.cluster_diag,
        ) as diag:
            for subj_idx, emg_idx in experiments:
                result = evaluate_optimization(...)
                diag.record_experiment(n_completed=1)
    """

    def __init__(
        self,
        tag: str = '',
        device: str = 'cpu',
        n_planned: int = 0,
        poll_interval_s: int = 30,
        enabled: bool = True,
    ) -> None:
        self._enabled = enabled
        self._tag = tag
        self._device = device
        self._n_planned = n_planned
        self._poll_interval_s = poll_interval_s

        self._t0: float = 0.0
        self._metrics = _DiagMetrics()
        self._poller: Optional[_GpuPoller] = None

    # ------------------------------------------------------------------
    # Context manager protocol
    # ------------------------------------------------------------------

    def __enter__(self) -> 'ClusterDiagnostics':
        if not self._enabled:
            return self

        self._t0 = time.time()
        m = self._metrics
        m.experiment_tag = self._tag
        m.n_experiments_planned = self._n_planned
        m.job_id = os.environ.get('SLURM_JOB_ID')
        m.array_task_id = os.environ.get('SLURM_ARRAY_TASK_ID')
        m.cluster_name = _detect_cluster()
        m.slurm_timelimit_s = _parse_slurm_timelimit(
            os.environ.get('SLURM_TIMELIMIT')
        )
        mem_mb = _parse_slurm_mem_mb()
        if mem_mb is not None:
            m.requested_mem_bytes = mem_mb * 1024 * 1024
            m.requested_ram_bytes = m.requested_mem_bytes
        raw_cpus = os.environ.get('SLURM_CPUS_PER_TASK', '').strip()
        if raw_cpus.isdigit():
            m.n_cpus_requested = int(raw_cpus)

        # GPU state
        try:
            import torch
            m.cuda_available = torch.cuda.is_available()
            if m.cuda_available:
                torch.cuda.reset_peak_memory_stats()
        except ImportError:
            pass

        # Start background GPU poller
        self._poller = _GpuPoller(interval_s=self._poll_interval_s)
        self._poller.start()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if not self._enabled:
            return False
        try:
            self._collect_metrics()
            print(self.format_terminal_report())
        except Exception:
            pass  # Diagnostics must never mask experiment exceptions
        return False  # Do not suppress caller exceptions

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_experiment(self, n_completed: int = 1) -> None:
        """Increment the completed-experiment counter.

        Call once per ``(subject, emg)`` pair after
        ``evaluate_optimization()`` (or equivalent) returns.

        Args:
            n_completed: Number of experiments just completed.
        """
        if self._enabled:
            self._metrics.n_experiments_completed += n_completed

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _collect_metrics(self) -> None:
        """Gather all metrics at job end.  Called from ``__exit__``."""
        m = self._metrics
        m.elapsed_s = time.time() - self._t0

        # Stop GPU poller
        if self._poller is not None:
            self._poller.stop()
            self._poller.join(timeout=5)
            m.gpu_util_samples = list(self._poller.samples)
            m.nvidia_smi_available = self._poller.available

        # GPU memory snapshot
        try:
            import torch
            if torch.cuda.is_available():
                m.peak_gpu_mem_bytes = torch.cuda.max_memory_allocated()
                m.reserved_gpu_mem_bytes = torch.cuda.memory_reserved()
        except (ImportError, RuntimeError):
            pass

        # CPU / RAM via resource module
        peak_rss, n_cpus = _collect_resource_metrics()
        m.peak_rss_bytes = peak_rss
        if n_cpus is not None and m.n_cpus_requested is None:
            m.n_cpus_requested = n_cpus

    def _compute_grade(self) -> str:
        """Compute weighted efficiency grade A–F.

        Sub-scores (each 0–1, linearly clamped):
          - GPU memory efficiency  (peak / requested):  40 pts
          - Walltime efficiency    (elapsed / timelimit): 30 pts
          - GPU utilisation        (mean / 100):         30 pts

        Grade thresholds: A ≥ 90, B ≥ 75, C ≥ 60, D ≥ 45, F < 45.
        Returns ``'?'`` when no sub-scores are available.

        Returns:
            Single uppercase letter ``'A'``–``'F'``, or ``'?'``.
        """
        score = 0.0
        total_weight = 0.0
        m = self._metrics

        if (m.cuda_available and m.requested_mem_bytes
                and m.requested_mem_bytes > 0 and m.peak_gpu_mem_bytes > 0):
            eff = min(1.0, m.peak_gpu_mem_bytes / m.requested_mem_bytes)
            score += 40.0 * eff
            total_weight += 40.0

        if m.slurm_timelimit_s and m.slurm_timelimit_s > 0:
            eff = min(1.0, m.elapsed_s / m.slurm_timelimit_s)
            score += 30.0 * eff
            total_weight += 30.0

        if m.gpu_util_samples:
            mean_util = sum(m.gpu_util_samples) / len(m.gpu_util_samples)
            score += 30.0 * min(1.0, mean_util / 100.0)
            total_weight += 30.0

        if total_weight == 0.0:
            return '?'

        normalised = score / total_weight * 100.0
        if normalised >= 90:
            return 'A'
        elif normalised >= 75:
            return 'B'
        elif normalised >= 60:
            return 'C'
        elif normalised >= 45:
            return 'D'
        return 'F'

    def _generate_warnings(self) -> List[Dict[str, Any]]:
        """Evaluate all warning rules and render text/fix templates.

        Returns:
            List of fired rule dicts, each augmented with
            ``'rendered_text'`` and ``'rendered_fix'`` keys.
        """
        m = self._metrics
        fired: List[Dict[str, Any]] = []

        for rule in _WARNING_RULES:
            try:
                if not rule['condition'](m):
                    continue
            except (ZeroDivisionError, TypeError, AttributeError):
                continue

            # Build template variable dict for this rule
            tv: Dict[str, Any] = {}

            if m.requested_mem_bytes and m.requested_mem_bytes > 0:
                tv['req_gb'] = m.requested_mem_bytes / (1024 ** 3)

            if m.requested_ram_bytes and m.requested_ram_bytes > 0:
                tv['req_ram_gb'] = m.requested_ram_bytes / (1024 ** 3)

            if m.peak_gpu_mem_bytes:
                tv['peak_gpu_gb'] = m.peak_gpu_mem_bytes / (1024 ** 3)
                tv['peak_gb'] = tv['peak_gpu_gb']
                if m.requested_mem_bytes:
                    tv['pct'] = m.peak_gpu_mem_bytes / m.requested_mem_bytes * 100
                    tv['suggested_mem_gb'] = max(
                        1.0, tv['peak_gpu_gb'] * 1.20
                    )

            if m.reserved_gpu_mem_bytes:
                tv['res_gb'] = m.reserved_gpu_mem_bytes / (1024 ** 3)
                if m.reserved_gpu_mem_bytes > 0 and m.peak_gpu_mem_bytes > 0:
                    tv['frag_pct'] = (
                        (m.reserved_gpu_mem_bytes - m.peak_gpu_mem_bytes)
                        / m.reserved_gpu_mem_bytes * 100
                    )

            if m.slurm_timelimit_s and m.slurm_timelimit_s > 0:
                tv['limit_h'] = m.slurm_timelimit_s / 3600.0
                tv['elapsed_h'] = m.elapsed_s / 3600.0
                tv['pct'] = m.elapsed_s / m.slurm_timelimit_s * 100
                # Suggest elapsed × 1.20, rounded up to nearest 15-min boundary
                sugg_s = m.elapsed_s * 1.20
                sh = int(sugg_s // 3600)
                sm = int((sugg_s % 3600) // 60)
                sm = ((sm + 14) // 15) * 15
                if sm >= 60:
                    sh += 1
                    sm = 0
                tv['suggested_time'] = f'{sh:02d}:{sm:02d}:00'

            if m.gpu_util_samples:
                tv['mean_util'] = sum(m.gpu_util_samples) / len(m.gpu_util_samples)

            if m.peak_rss_bytes:
                tv['peak_rss_gb'] = m.peak_rss_bytes / (1024 ** 3)
                if m.requested_ram_bytes and m.requested_ram_bytes > 0:
                    tv['pct'] = m.peak_rss_bytes / m.requested_ram_bytes * 100
                    tv['suggested_mem_gb'] = max(1.0, tv['peak_rss_gb'] * 1.25)

            if m.n_cpus_requested is not None:
                tv['n_cpus'] = m.n_cpus_requested

            try:
                r_text = rule['text'].format(**tv)
                r_fix = rule['fix'].format(**tv)
            except KeyError:
                r_text = rule['text']
                r_fix = rule['fix']

            fired.append({**rule, 'rendered_text': r_text, 'rendered_fix': r_fix})

        return fired

    def format_terminal_report(self) -> str:
        """Render the terminal summary as a 72-char-wide ASCII box string.

        Traffic lights in each metric row:
          ``[OK]`` — at or above soft threshold
          ``[! ]`` — between soft and hard threshold (warning)
          ``[X ]`` — below hard threshold (critical)
          ``[--]`` — metric not applicable (e.g. no CUDA)
          ``[??]`` — SLURM env var absent

        Returns:
            Multi-line string suitable for ``print()``.
        """
        m = self._metrics
        W = 72          # total box width
        IW = W - 2      # inner width (between │ characters)

        def _row(text: str = '') -> str:
            return f"|{text:<{IW}}|"

        def _sep(c: str = '-') -> str:
            return f"+{c * IW}+"

        def _tl(val: float, soft: float, hard: float,
                higher_is_better: bool = True) -> str:
            """1-char traffic light: OK / !  / X."""
            if higher_is_better:
                if val >= soft:
                    return 'OK'
                return '! ' if val >= hard else 'X '
            else:
                if val <= soft:
                    return 'OK'
                return '! ' if val <= hard else 'X '

        lines = [_sep('=')]
        header = f'  CLUSTER DIAGNOSTICS  —  {m.experiment_tag}'
        lines.append(_row(header))
        cluster_part = m.cluster_name.upper()
        job_part = m.job_id or 'N/A'
        if m.array_task_id:
            job_part += f'[{m.array_task_id}]'
        lines.append(_row(f'  Cluster: {cluster_part}  |  Job: {job_part}'))
        lines.append(_sep())

        # ---- Walltime ----
        if m.slurm_timelimit_s and m.slurm_timelimit_s > 0:
            eff = m.elapsed_s / m.slurm_timelimit_s
            tl = _tl(eff, 0.55, 0.30)
            lines.append(_row(
                f'  [{tl}] Walltime : '
                f'{m.elapsed_s/3600:.2f}h / {m.slurm_timelimit_s/3600:.2f}h req  '
                f'({eff*100:.0f}%)'
            ))
        else:
            lines.append(_row(
                f'  [??] Walltime : {m.elapsed_s/3600:.2f}h  '
                f'(SLURM_TIMELIMIT not set)'
            ))

        # ---- GPU Memory ----
        if m.cuda_available:
            peak_gb = m.peak_gpu_mem_bytes / (1024 ** 3)
            res_gb = m.reserved_gpu_mem_bytes / (1024 ** 3)
            if m.requested_mem_bytes and m.requested_mem_bytes > 0:
                req_gb = m.requested_mem_bytes / (1024 ** 3)
                eff = m.peak_gpu_mem_bytes / m.requested_mem_bytes
                tl = _tl(eff, 0.75, 0.40)
                lines.append(_row(
                    f'  [{tl}] GPU Mem  : '
                    f'{peak_gb:.2f} GB peak / {req_gb:.2f} GB req  '
                    f'({eff*100:.0f}%)'
                ))
            else:
                lines.append(_row(
                    f'  [??] GPU Mem  : {peak_gb:.2f} GB peak  '
                    f'(SLURM_MEM_PER_NODE not set)'
                ))
            if res_gb > 0:
                frag = (res_gb - peak_gb) / res_gb * 100 if res_gb > 0 else 0
                tl = _tl(frag, 25.0, 50.0, higher_is_better=False)
                lines.append(_row(
                    f'  [{tl}] CUDA Frag: reserved {res_gb:.2f} GB, '
                    f'frag {frag:.0f}%'
                ))
        else:
            lines.append(_row('  [--] GPU Mem  : CUDA not available on this node'))

        # ---- GPU Utilisation ----
        if m.nvidia_smi_available and m.gpu_util_samples:
            mean_u = sum(m.gpu_util_samples) / len(m.gpu_util_samples)
            tl = _tl(mean_u, 50.0, 20.0)
            lines.append(_row(
                f'  [{tl}] GPU Util : {mean_u:.0f}% mean  '
                f'(n={len(m.gpu_util_samples)} polls)'
            ))
        elif not m.nvidia_smi_available:
            lines.append(_row('  [--] GPU Util : nvidia-smi not available'))

        # ---- CPU / RAM ----
        if m.peak_rss_bytes > 0 and m.requested_ram_bytes and m.requested_ram_bytes > 0:
            rss_gb = m.peak_rss_bytes / (1024 ** 3)
            req_gb = m.requested_ram_bytes / (1024 ** 3)
            eff = m.peak_rss_bytes / m.requested_ram_bytes
            tl = _tl(eff, 0.70, 0.40)
            lines.append(_row(
                f'  [{tl}] Peak RSS : '
                f'{rss_gb:.2f} GB / {req_gb:.2f} GB req  '
                f'({eff*100:.0f}%)'
            ))
        elif m.peak_rss_bytes > 0:
            rss_gb = m.peak_rss_bytes / (1024 ** 3)
            lines.append(_row(f'  [??] Peak RSS : {rss_gb:.2f} GB  (no SLURM_MEM_PER_NODE)'))
        else:
            lines.append(_row('  [--] Peak RSS : unavailable (non-Linux or resource module absent)'))

        if m.n_cpus_requested is not None:
            lines.append(_row(f'  [--] CPUs     : {m.n_cpus_requested} requested (SLURM_CPUS_PER_TASK)'))

        # ---- Throughput ----
        lines.append(_sep())
        elapsed_h = m.elapsed_s / 3600.0
        if elapsed_h > 0 and m.n_experiments_completed > 0:
            cuda_vis = os.environ.get('CUDA_VISIBLE_DEVICES', '')
            n_gpus = max(1, len([x for x in cuda_vis.split(',') if x.strip()])) \
                if cuda_vis.strip() else 1
            tput = m.n_experiments_completed / (elapsed_h * n_gpus)
            lines.append(_row(
                f'  [--] Throughput: {tput:.2f} experiments/GPU-hour  '
                f'({m.n_experiments_completed}/{m.n_experiments_planned} done)'
            ))
        else:
            lines.append(_row(
                f'  [--] Throughput: '
                f'{m.n_experiments_completed}/{m.n_experiments_planned} experiments done'
            ))

        # ---- Grade ----
        grade = self._compute_grade()
        _icons = {'A': '***', 'B': '** ', 'C': '*  ', 'D': '   ', 'F': '!!!', '?': '???'}
        icon = _icons.get(grade, '   ')
        lines.append(_sep('='))
        lines.append(_row(f'  {icon}  EFFICIENCY GRADE: {grade}  {icon}'))
        lines.append(_sep('='))

        # ---- Warnings ----
        warnings = self._generate_warnings()
        if warnings:
            lines.append(_row('  WARNINGS:'))
            for w in warnings:
                lines.append(_row())
                for wline in _wrap_lines(w['rendered_text'], IW - 6):
                    lines.append(_row(f'  !! {wline}'))
                lines.append(_row('  FIX:'))
                for fline in w['rendered_fix'].splitlines():
                    lines.append(_row(f'    {fline}'))
            lines.append(_sep())
        else:
            lines.append(_row('  No warnings — well-tuned job!'))
            lines.append(_sep())

        return '\n'.join(lines)
