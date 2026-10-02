"""Per-job efficiency table from `mila.sh report` / `narval.sh report` output.

    (bash scripts/mila.sh report 2026-10-01; bash scripts/narval.sh report 2026-10-01) | python scripts/cluster_report.py

Each cluster section holds the raw `sacct -P` rows and the key lines of every job log's CLUSTER DIAGNOSTICS block
(grade, GPU memory, GPU utilisation, throughput). The table joins them per job and scores each one. The score is
the mean of the components that could be measured, each capped at 1 against the healthy band of the rubric in
``.claude/skills/cluster-companion/references/slurm_primer.md`` (CPU 85%, memory 60-90%, time 50%, GPU utilisation 70%),
so a job is only marked down for what the rubric says to act on.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from typing import Optional

#: Healthy-band edges (fractions); mirror HEALTHY_BAND_LOW in src/pfns4neurostim/diagnostics/cluster.py.
CPU_TARGET: float = 0.85
MEM_LOW: float = 0.60
MEM_HIGH: float = 0.90
TIME_TARGET: float = 0.50
GPU_UTIL_TARGET: float = 0.70
#: Score of a job whose memory use is above the band: no waste, but one step from an OOM kill.
MEM_HIGH_SCORE: float = 0.5
#: Memory units of `sacct` MaxRSS / ReqMem, in GB.
_UNIT_GB = {'K': 1 / 1024 ** 2, 'M': 1 / 1024, 'G': 1.0, 'T': 1024.0}
#: Slurm states that are failures to report ahead of any efficiency flag.
FAILURE_STATES = ('OUT_OF_MEMORY', 'TIMEOUT', 'FAILED', 'NODE_FAIL')

SACCT_FIELDS = ('JobID', 'JobName', 'State', 'Elapsed', 'Timelimit', 'AllocCPUS', 'ReqMem', 'TotalCPU',
                'CPUTime', 'MaxRSS', 'AllocTRES', 'ExitCode')


@dataclass
class Job:
    """One Slurm job joined with its diagnostics block."""

    cluster: str
    job_id: str
    name: str = ''
    state: str = ''
    elapsed_s: float = 0.0
    limit_s: Optional[float] = None
    cpus: int = 0
    req_mem_gb: Optional[float] = None
    total_cpu_s: float = 0.0
    cpu_time_s: float = 0.0
    max_rss_gb: Optional[float] = None
    gpus: int = 0
    grade: str = ''
    gpu_mem: str = ''
    gpu_util: Optional[float] = None
    throughput: str = ''
    flags: list[str] = field(default_factory=list)

    @property
    def cpu_eff(self) -> Optional[float]:
        """TotalCPU / CPUTime: the share of the reserved cores that did work."""
        return self.total_cpu_s / self.cpu_time_s if self.cpu_time_s > 0 else None

    @property
    def mem_eff(self) -> Optional[float]:
        """MaxRSS / ReqMem (MaxRSS is the batch step's, so with lanes it is the sum over lanes)."""
        if self.max_rss_gb is None or not self.req_mem_gb:
            return None
        return self.max_rss_gb / self.req_mem_gb

    @property
    def time_eff(self) -> Optional[float]:
        """Elapsed / Timelimit."""
        return self.elapsed_s / self.limit_s if self.limit_s else None


def parse_duration(text: str) -> Optional[float]:
    """Seconds in a Slurm duration (`D-HH:MM:SS`, `HH:MM:SS`, `MM:SS.mmm`), or None when blank or unlimited."""
    text = text.strip()
    if not text or text in ('UNLIMITED', 'Partition_Limit', 'INVALID'):
        return None
    days = 0
    if '-' in text:
        d, text = text.split('-', 1)
        days = int(d)
    parts = [float(p) for p in text.split(':')]
    if len(parts) == 2:
        parts = [0.0] + parts          # TotalCPU prints MM:SS.mmm
    h, m, s = parts
    return days * 86400 + h * 3600 + m * 60 + s


def parse_gb(text: str) -> Optional[float]:
    """GB in a Slurm memory string such as `7G`, `3Gn`, `1704264K`; None when blank."""
    m = re.fullmatch(r'\s*([0-9.]+)([KMGT])[nc]?\s*', text)
    return float(m.group(1)) * _UNIT_GB[m.group(2)] if m else None


def parse_report(raw: str) -> list[Job]:
    """Join the sacct rows and diagnostics lines of every cluster section in `raw` into Job records."""
    jobs: dict[tuple[str, str], Job] = {}
    cluster, section = 'unknown', 'sacct'
    for line in raw.splitlines():
        if line.startswith('##CLUSTER'):
            cluster, section = line.split()[1], 'sacct'
        elif line.startswith('##DIAG'):
            section = 'diag'
        elif section == 'sacct' and line.count('|') >= len(SACCT_FIELDS) - 1:
            row = dict(zip(SACCT_FIELDS, line.split('|')))
            base = row['JobID'].split('.')[0]
            job = jobs.setdefault((cluster, base), Job(cluster=cluster, job_id=base))
            if '.' not in row['JobID']:
                job.name, job.state = row['JobName'], row['State'].split()[0]
                job.elapsed_s = parse_duration(row['Elapsed']) or 0.0
                job.limit_s = parse_duration(row['Timelimit'])
                job.cpus = int(row['AllocCPUS'] or 0)
                job.req_mem_gb = parse_gb(row['ReqMem'])
                job.total_cpu_s = parse_duration(row['TotalCPU']) or 0.0
                job.cpu_time_s = parse_duration(row['CPUTime']) or 0.0
                gres = re.search(r'gres/gpu=(\d+)', row['AllocTRES'])
                job.gpus = int(gres.group(1)) if gres else 0
            rss = parse_gb(row['MaxRSS'])
            if rss is not None:
                job.max_rss_gb = max(rss, job.max_rss_gb or 0.0)
        elif section == 'diag':
            m = re.match(r'\./\w*?_(\d+)\.out:(.*)', line)     # lane0_<id>.out, stress_<id>.out, cpu_<id>.out, ...
            if not m or (cluster, m.group(1)) not in jobs:
                continue
            job, text = jobs[(cluster, m.group(1))], m.group(2)
            if (g := re.search(r'EFFICIENCY GRADE:\s*([A-F?])', text)):
                job.grade = g.group(1)
            elif (gm := re.search(r'GPU Mem\s*:.*?([0-9.]+) GB.*\((\d+)%\)', text)):
                job.gpu_mem = f'{float(gm.group(1)):.2f}GB {gm.group(2)}%'   # job peak and its share of the card / request
            elif (u := re.search(r'GPU Util\s*:\s*(\d+)%', text)):
                job.gpu_util = float(u.group(1)) / 100
            elif (t := re.search(r'Throughput:\s*([0-9.]+) experiments/GPU-hour', text)):
                job.throughput = t.group(1)
    return list(jobs.values())


def score(job: Job) -> Optional[float]:
    """Mean of the measurable components, each capped at 1 against its healthy band, as a 0-100 score."""
    parts: list[float] = []
    if job.cpu_eff is not None and job.cpus > 1:
        parts.append(min(1.0, job.cpu_eff / CPU_TARGET))
    if job.mem_eff is not None:
        e = job.mem_eff
        parts.append(min(1.0, e / MEM_LOW) if e <= MEM_HIGH else MEM_HIGH_SCORE)
    if job.time_eff is not None:
        parts.append(min(1.0, job.time_eff / TIME_TARGET))
    if job.gpu_util is not None:
        parts.append(min(1.0, job.gpu_util / GPU_UTIL_TARGET))
    return 100 * sum(parts) / len(parts) if parts else None


def flag(job: Job) -> list[str]:
    """Short action flags: failures first, then what the rubric says to change."""
    out: list[str] = []
    done = job.state == 'COMPLETED'
    if job.state in FAILURE_STATES:
        out.append(job.state)
    elif job.state.startswith('CANCELLED') and job.elapsed_s == 0:
        out.append('never-started')
    if job.mem_eff is not None and job.mem_eff > MEM_HIGH:
        out.append('mem-tight')
    elif job.mem_eff is not None and job.mem_eff < MEM_LOW and done:
        out.append('mem-over')
    if job.time_eff is not None and job.time_eff < TIME_TARGET and done:
        out.append('time-over')
    if job.cpu_eff is not None and job.cpus > 1 and job.cpu_eff < CPU_TARGET and done:
        out.append('cpu-idle')
    if job.gpu_util is not None and job.gpu_util < GPU_UTIL_TARGET:
        out.append('gpu-idle')
    return out


def _pct(x: Optional[float]) -> str:
    return '-' if x is None else f'{100 * x:.0f}%'


def _hms(s: Optional[float]) -> str:
    return '-' if s is None else f'{int(s // 3600)}:{int(s % 3600 // 60):02d}'


def render(jobs: list[Job]) -> str:
    """The table, one row per job that ran, followed by per-cluster totals."""
    head = (f"{'cluster':7} {'job':>9} {'name':14} {'state':11} {'run':>6} {'limit':>6} {'time':>5} {'cpu':>5} "
            f"{'mem':>5} {'gpu mem':>9} {'gpu util':>8} {'exp/GPUh':>8} {'score':>5} {'gr':>2}  flags")
    lines = [head, '-' * len(head)]
    for job in sorted(jobs, key=lambda j: (j.cluster, j.job_id)):
        if job.elapsed_s == 0:
            continue                                   # withdrawn or never-started rows carry no efficiency data
        sc = score(job)
        lines.append(
            f"{job.cluster:7} {job.job_id:>9} {job.name[:14]:14} {job.state[:11]:11} {_hms(job.elapsed_s):>6} "
            f"{_hms(job.limit_s):>6} {_pct(job.time_eff):>5} {_pct(job.cpu_eff):>5} {_pct(job.mem_eff):>5} "
            f"{job.gpu_mem[:9]:>9} {_pct(job.gpu_util):>8} {job.throughput or '-':>8} "
            f"{'-' if sc is None else f'{sc:.0f}':>5} {job.grade or '-':>2}  {' '.join(flag(job))}"
        )
    for cluster in sorted({j.cluster for j in jobs}):
        mine = [j for j in jobs if j.cluster == cluster and j.elapsed_s > 0]
        scored = [s for j in mine if (s := score(j)) is not None]
        cpu_h = sum(j.cpus * j.elapsed_s for j in mine) / 3600
        gpu_h = sum(j.gpus * j.elapsed_s for j in mine) / 3600
        summary = f"{cluster}: {len(mine)} jobs, {cpu_h:.1f} CPU-h, {gpu_h:.1f} GPU-h"
        lines.append(summary + (f", mean score {sum(scored) / len(scored):.0f}" if scored else ', no scored jobs'))
    return '\n'.join(lines)


def main() -> None:
    """Read the combined `report` output on stdin and print the table."""
    jobs = parse_report(sys.stdin.read())
    if not jobs:
        sys.exit('cluster_report: no jobs found in the input (did `report` print a ##CLUSTER section?)')
    print(render(jobs))


if __name__ == '__main__':
    main()
