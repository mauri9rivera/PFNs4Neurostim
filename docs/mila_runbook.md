# Mila runbook

Every number in the paper and the deck comes from Mila (the Windows dev box has a
duplicate-OpenMP defect and is smoke-test only). The agent is **read-only** on the cluster:
**you run every `sbatch`**; the agent reads queue state and logs over an SSH control socket.

## 1. Supervised access (`scripts/mila.sh`)

Mila asks for an OTP on every SSH connection, so no unattended connection is possible. You
authenticate once and the agent reuses that connection.

```bash
bash scripts/mila.sh open          # prints the command YOU run (asks for the OTP once)
bash scripts/mila.sh status        # socket live?
bash scripts/mila.sh queue         # squeue --me
bash scripts/mila.sh logs <jobid> --tail 200
bash scripts/mila.sh sacct <jobid>
bash scripts/mila.sh run -- 'bash scripts/mila_setup.sh verify'
bash scripts/mila.sh close
```

* `run` refuses anything that mutates state (`sbatch`, `scancel`, `rm`, `pip`, redirection, ...)
  and fails within seconds if the socket is down, instead of hanging on a hidden OTP prompt.
* `submit <script> <config>` only **prints** the `sbatch` line for you to run.
* Socket path: `MILA_SOCKET` (default `~/.ssh/cm-mila.sock`); host alias: `MILA_HOST` (default `mila`).

## 2. Deployment steps (run on a login node unless noted)

```bash
# 0. (you, once per 8 h) open the SSH control socket so the agent can read queue/logs: bash scripts/mila.sh open
# 1. update the code
cd ~/projects/PFNs4Neurostim
git checkout scripts/mila_setup.sh && git pull          # drop the hand-copied file first, then pull
bash scripts/mila_setup.sh install                      # editable install into the MAIN env
# 2. bench env (Python 3.11: TabICL, TabFM) - ~10-20 min, only needed for the D3 PFN benchmark
sbatch scripts/setup_env_job.sh bench                 # NOT on the login node: conda is killed there (memory limit)
bash scripts/mila_setup.sh submodules                   # libs/tabicl + libs/tabfm are used from the submodules via sys.path
# 3. smoke-test the bench env on a login node (no GPU needed for the import checks)
module load anaconda/3 && conda run -n pfns4neurostim-bench python -c "from pfns4neurostim.models.pfn.external import availability; print(availability())"
# 4. calibrate TabFM (unmeasured, GPU only): 2 reps of one channel
CONDA_ENV=pfns4neurostim-bench sbatch scripts/run_bo_benchmark.sh configs/experiment/hyp0_pfn_bench_nhp.yaml "models=[tabfm]" dataset.subjects=[1] dataset.emgs=[0] n_reps=2
# 5. submit every deliverable but spinal (core), then the externals when you choose (TabICL, TabFM, PFNs4BO)
bash scripts/submit_portfolio.sh
bash scripts/submit_externals.sh
```

`python scripts/portfolio.py` prints the same plan with wall-time estimates. The two submit scripts are generated from it
(`--emit-bash` for `core`, `--emit-bash --group externals`); edit `scripts/portfolio.py`, not the generated files.

## 3. Which environment runs what

| Experiment | Env | Why |
|---|---|---|
| `bo_benchmark` with TabPFN-2.5 / GP / Random (`hyp_a_*`, `hyp0_acq_table_*`) | `pfns4neurostim` (3.9) | pinned stack |
| `stress_sweep` (K2, K5, K6) | `pfns4neurostim` (3.9) | pinned stack |
| `bo_benchmark` with TabICL v2 / TabFM (`hyp0_pfn_bench_*`) | `pfns4neurostim-bench` (3.11) | TabICL needs >= 3.10, TabFM >= 3.11 (TabFlex dropped: dead weight host) |
| TabPFN v1, Mitra | not yet | deferred (own env / AutoGluon) |

`scripts/run_*.sh` select the env with `CONDA_ENV` (default `pfns4neurostim`).

## 4. Submitting (you run these)

`python scripts/portfolio.py` prints every job in priority order with wall-time estimates; it only prints.

**Multi-process lanes.** Per-user caps on Mila are 2 GPUs + 8 CPUs + 48 GB on `main`, and a separate 8 CPUs + 64 GB on
`main-cpu`. TabPFN uses ~5% of a GPU and ~1.3 GB of RAM, so one job runs several processes ("lanes"), each owning every
N-th channel (`--shard i/N`); the GP models need no GPU and run on `main-cpu`, one lane per core.

```bash
cd ~/projects/PFNs4Neurostim
# GPU job: 4 lanes share one GPU (4 CPUs, 10 GB)
LANES=4 sbatch scripts/run_bo_benchmark.sh configs/experiment/hyp_a_5d_rat.yaml "models=[tabpfn_v2_5]"
LANES=4 sbatch scripts/run_stress_sweep.sh configs/experiment/stress_k2_channel_nhp.yaml "models=[tabpfn_v2_5]"
# CPU job: GP models + random on main-cpu, 8 single-thread lanes (device=cpu is forced)
sbatch scripts/run_cpu.sh bo_benchmark configs/experiment/hyp_a_5d_rat.yaml "models=[gp_mll,gp_naive,random]"
sbatch scripts/run_cpu.sh stress_sweep configs/experiment/stress_k2_channel_nhp.yaml "models=[gp_mll,gp_naive]"
# bench env (TabICL / TabFlex): select it with CONDA_ENV
CONDA_ENV=pfns4neurostim-bench LANES=4 sbatch scripts/run_bo_benchmark.sh configs/experiment/hyp0_pfn_bench_nhp.yaml "models=[tabpfn_v2_5,tabicl,tabflex]"
```

Lane logs are `logs/lane<i>_<jobid>.out` (`bash scripts/mila.sh logs <jobid>` reads them together with the job log). All jobs and lanes of an
experiment share ONE run dir `<family>-<tag>`: they run `--compute-only`, so each writes only its cells (to the shared cache) and a
provenance record `shards/<node>-job<id>-shard<i>of<N>-<models>.json` (node, GPU model, CPU model, models, devices, cell counts).
The assemble job (`scripts/run_assemble.sh`, submitted automatically with `--dependency=afterany`) writes `tidy.csv`, tables and
figures from the union and embeds the compact registry in `config.yaml` under `shards:`; every tidy row also carries
`host_node`, `host_gpu`, `host_cpu`. To assemble by hand (login node or locally):

```bash
python -m pfns4neurostim bo_benchmark --config configs/experiment/hyp_a_5d_rat.yaml --only-cached
```

A failed lane does not stop the others; its finished cells are already cached, and re-submitting the same command computes only
what is missing. TabFlex downloads its weights on first use: run one short TabFlex fit on a **login node** first, since a compute
node may have no outbound network.

**Resume.** Every finished cell is written to `output/cells/` as it completes. Jobs carry `--requeue --signal=B:TERM@300`: on
preemption or 5 minutes before the time limit the lanes stop, the job requeues itself, and resumes from the cache. Re-running any
command is safe and only computes missing cells. `--no-cache` ignores the cache; `--only-cached` assembles outputs from cached
cells without computing.

## 5. Collecting results (run locally)

```bash
wsl -e bash -lc 'cd /mnt/c/workspace/PFNs4Neurostim && bash scripts/export_results.sh'   # benchmark/, stress/, cells/ into the SAME output/ tree
# never overwrites local data: cells use --ignore-existing; other files only if newer, the replaced file goes to output/.pull_backup/
python -m pfns4neurostim stress_sweep --config configs/experiment/stress_k2_channel_nhp.yaml --replot
```

Run directories are `output/benchmark/<dataset>/<family>-<tag>/` and
`output/stress/<knob>/<dataset>/<family>-<tag>/`.

## 6. Failure modes (observed 2026-09-20 by reading the real SLURM logs)

| Symptom (in `logs/stress_<jobid>.err`) | Cause | Fix |
|---|---|---|
| `conda.sh: line 55: PS1: unbound variable` (jobs 10854576, 10854578) | `module load` / `conda activate` read unset variables and abort under `set -u` | `job_activate` in `scripts/_job_common.sh` wraps them in `set +u ... set -u` (the cluster copy of `run_stress_sweep.sh` had been hand-patched the same way) |
| `ModuleNotFoundError: No module named 'pfns4neurostim.data'` (jobs 10854595, 10854603) | `.gitignore` had `data` / `data/`, which also matched `src/pfns4neurostim/data/` and `tests/data/`, so the package was **never committed** and the cluster checkout lacks it | `.gitignore` now anchors `/data` and `/data/`. Commit, push, then on the cluster `git checkout scripts/run_stress_sweep.sh && git pull` (drop the local hand-patch first) and re-run `bash scripts/mila_setup.sh install` |
| `conda: command not found` from `bash scripts/mila.sh run -- 'bash scripts/mila_setup.sh verify'` | a non-interactive SSH shell has no conda on `PATH` | `mila_setup.sh` now calls `load_conda` (loads `anaconda/3`) in `env`, `install`, `deps`, `verify` |
| `set: pipefail: invalid option name` | a shell script with CRLF line endings (Windows edit) | `.gitattributes` forces `*.sh` to LF; run `sed -i 's/\r$//' scripts/*.sh` if a working copy was edited on Windows |
| `Killed` from `conda env create` on the login node | the login node's per-process memory limit is hit parsing the conda-forge index | build inside a job: `sbatch scripts/setup_env_job.sh bench` |
| `Permission denied (keyboard-interactive)` from `mila.sh run` | control socket not open | `bash scripts/mila.sh open`, run the printed command, enter the OTP |

Other state observed on the cluster: the checkout is at commit `1bf4a0a` (stale); the main env
`pfns4neurostim` exists (torch 2.5.1+cu118); **`pfns4neurostim-bench` does not exist yet**; data
present: `monkeys/` (4 subjects) and `5d_rat/` (6 subjects), no `rat/` or `spinal/`; `$SCRATCH` holds
only 372 MB. A stray `slurm-10848085.out` in the repo root is from an interactive allocation that
hit its 30-minute limit, not from a sweep.

### Using the socket from WSL

The control socket was opened from WSL, so it lives in the WSL filesystem. From Windows, call the
script through `wsl`: `wsl -e bash -lc 'cd /mnt/c/workspace/PFNs4Neurostim && bash scripts/mila.sh status'`.
