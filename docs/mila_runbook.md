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

## 2. Layout and setup (run on a login node)

```bash
git clone --recurse-submodules https://github.com/mauri9rivera/PFNs4Neurostim.git ~/projects/PFNs4Neurostim
cd ~/projects/PFNs4Neurostim
bash scripts/mila_setup.sh layout       # $SCRATCH data/output/logs + repo symlinks
bash scripts/mila_setup.sh env          # MAIN env  (pfns4neurostim, Python 3.9)
bash scripts/mila_setup.sh env bench    # BENCH env (pfns4neurostim-bench, Python 3.11)
bash scripts/mila_setup.sh submodules   # libs/tabicl, libs/ticl, libs/tabfm (+ local exclude for weights)
bash scripts/mila_setup.sh verify
```

Code lives on `$HOME`, working data and outputs on `$SCRATCH` (purged after 90 days without
access: `bash scripts/mila_setup.sh touch`), and the raw-data master on `$ARCHIVE`.

## 3. Which environment runs what

| Experiment | Env | Why |
|---|---|---|
| `bo_benchmark` with TabPFN-2.5 / GP / Random (`hyp_a_*`, `hyp0_acq_table_*`) | `pfns4neurostim` (3.9) | pinned stack |
| `stress_sweep` (K2, K5, K6) | `pfns4neurostim` (3.9) | pinned stack |
| `bo_benchmark` with TabICL v2 / TabFlex / TabFM (`hyp0_pfn_bench_*`) | `pfns4neurostim-bench` (3.11) | TabICL needs >= 3.10, TabFM >= 3.11, TabFlex pulls wandb/mlflow |
| TabPFN v1, Mitra | not yet | deferred (own env / AutoGluon) |

`scripts/run_*.sh` select the env with `CONDA_ENV` (default `pfns4neurostim`).

## 4. Submitting (you run these)

```bash
cd ~/projects/PFNs4Neurostim
sbatch scripts/run_bo_benchmark.sh configs/experiment/hyp_a_nhp.yaml n_reps=5 dataset.emgs=[0,1,2]
sbatch scripts/run_stress_sweep.sh configs/experiment/stress_k2_nhp.yaml n_reps=5
CONDA_ENV=pfns4neurostim-bench sbatch --export=ALL scripts/run_bo_benchmark.sh configs/experiment/hyp0_pfn_bench_nhp.yaml n_reps=5
```

TabFlex downloads its weights on first use: run one short TabFlex fit on a **login node**
first, since a compute node may have no outbound network.

**Resume.** Every finished cell is written to `output/cells/` as it completes. Jobs carry
`--requeue --signal=B:TERM@300`: on preemption or 5 minutes before the time limit the job
stops, requeues itself, and resumes from the cache. Re-running any command is safe and only
computes missing cells. `--no-cache` ignores the cache; `--only-cached` assembles outputs from
cached cells without computing.

## 5. Collecting results (run locally)

```bash
bash scripts/export_results.sh <mila_username>            # benchmark/, stress/, cells/
python -m pfns4neurostim stress_sweep --config configs/experiment/stress_k2_nhp.yaml --replot
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
| `Permission denied (keyboard-interactive)` from `mila.sh run` | control socket not open | `bash scripts/mila.sh open`, run the printed command, enter the OTP |

Other state observed on the cluster: the checkout is at commit `1bf4a0a` (stale); the main env
`pfns4neurostim` exists (torch 2.5.1+cu118); **`pfns4neurostim-bench` does not exist yet**; data
present: `monkeys/` (4 subjects) and `5d_rat/` (6 subjects), no `rat/` or `spinal/`; `$SCRATCH` holds
only 372 MB. A stray `slurm-10848085.out` in the repo root is from an interactive allocation that
hit its 30-minute limit, not from a sweep.

### Using the socket from WSL

The control socket was opened from WSL, so it lives in the WSL filesystem. From Windows, call the
script through `wsl`: `wsl -e bash -lc 'cd /mnt/c/workspace/PFNs4Neurostim && bash scripts/mila.sh status'`.
