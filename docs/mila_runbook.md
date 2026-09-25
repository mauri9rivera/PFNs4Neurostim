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
# 2b. v1 env (tabpfn<2: TabPFN v1) - only needed for portfolio units E7/E8
sbatch scripts/setup_env_job.sh v1
bash scripts/mila_setup.sh submodules                   # libs/tabicl + libs/tabfm are used from the submodules via sys.path
# 3. smoke-test the bench env on a login node (no GPU needed for the import checks)
module load anaconda/3 && conda run -n pfns4neurostim-bench python -c "from pfns4neurostim.models.pfn.external import availability; print(availability())"
#    the v1 env must report tabpfn_v1 True and tabpfn_v2_5's backend absent - if it reports the opposite, tabpfn 6.3.2 leaked in
module load anaconda/3 && conda run -n pfns4neurostim-v1 python -c "from pfns4neurostim.models.pfn.external import availability; print(availability())"
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
| `bo_benchmark` with TabPFN v1 (`hyp0_pfn_bench_*`, units E7/E8) | `pfns4neurostim-v1` | `tabpfn<2` and the pinned `tabpfn` 6.3.2 own the same module name |
| Mitra | not yet | deferred (AutoGluon) |

`scripts/run_*.sh` select the env with `CONDA_ENV` (default `pfns4neurostim`). The authoritative
record of model -> env is `ExternalSpec.env` in `models/pfn/external.py`, not this table; an
availability failure names the env to activate.

## 3b. Replacing a dataset cohort (5d_rat, 2026-09-25)

`data/5d_rat` moved to the `noOutliers` cohort: one directory per animal, validity from the lab's
own `valid_own` / `flag_valid_emg`, `rData03` dropped (so **every subject index below it shifted
down by one**). Every 5d_rat number computed before that date is stale.

As of 2026-09-25 the `$ARCHIVE` master holds no `data/` at all and `$SCRATCH/pfns4neurostim/data/5d_rat`
holds the OLD flat files, so the new cohort is uploaded from your machine (WSL, reusing the control socket),
to the ARCHIVE master and to the working copy:

```bash
wsl -e bash -lc 'cd /mnt/c/workspace/PFNs4Neurostim && rsync -a -e "ssh -S $HOME/.ssh/cm-mila.sock" data/5d_rat/ mila:/network/archive/m/mauricio.rivera/pfns4neurostim/data/5d_rat/'
wsl -e bash -lc 'cd /mnt/c/workspace/PFNs4Neurostim && rsync -a -e "ssh -S $HOME/.ssh/cm-mila.sock" data/5d_rat/ mila:/network/scratch/m/mauricio.rivera/pfns4neurostim/data/5d_rat/'
```

Then on the login node, from the repo root, after `git pull`:

```bash
bash scripts/clean_mila_results.sh
bash scripts/clean_mila_results.sh --apply
```

`clean_mila_results.sh` checks that this checkout carries the cohort stamp and that every animal directory
of the new cohort is present (it refuses `--apply` otherwise), then (1) moves the old flat files out of
`data/5d_rat` into `data/_retired/`, (2) runs `retire_dataset_cohort.py` (stale cells deleted, run data
quarantined, figures left as a proxy) and (3) runs `prune_results.py` (results the code can no longer
request, archived under `output/archive/<stamp>-pruned/`). Steps 2-3 are cleanup, not correctness: the
cohort stamp is part of every 5d_rat cell's identity, so a stale cell can never be returned as a hit.
Every step is idempotent, and the dry run comes first on purpose.

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
present: `monkeys/` (4 subjects) and `5d_rat/` (the pre-2026-09-25 cohort, 6 subjects — restage it for the
5-subject `noOutliers` cohort), no `rat/` or `spinal/`; `$SCRATCH` holds
only 372 MB. A stray `slurm-10848085.out` in the repo root is from an interactive allocation that
hit its 30-minute limit, not from a sweep.

### Using the socket from WSL

The control socket was opened from WSL, so it lives in the WSL filesystem. From Windows, call the
script through `wsl`: `wsl -e bash -lc 'cd /mnt/c/workspace/PFNs4Neurostim && bash scripts/mila.sh status'`.
