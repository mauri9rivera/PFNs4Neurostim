#!/bin/bash
# K2-channel on NHP, PFN half, sharded over both GPUs of the per-user cap (2 GPUs on `main`).
#
# Each job runs 4 single-threaded lanes (`--shard i/4`) on its own GPU over a disjoint set of subjects, so the two never touch the same
# channel: NHP has 18 channels (subject 0: 6, subject 1: 8, subject 3: 4), split 8 + 10. Both write to the shared cell cache under the
# config's tag, so a requeue or a rerun resumes from the cells already finished.
#
# The GP half (gp_mll, gp_naive) is NOT submitted here: it runs locally, and the final tables come from one `--only-cached` assembly on
# the full config (no subject override) after the cells are pulled home.
#
# The agent never submits: you run this ONCE on the Mila login node, from the repo root.
#   bash scripts/submit_k2_nhp.sh
#
# Env overrides: CONFIG, MODELS, LANES, MEM, TIME, SUBJECTS_A, SUBJECTS_B, CONDA_ENV.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

CONFIG="${CONFIG:-configs/experiment/stress_k2_channel_nhp.yaml}"
MODELS="${MODELS:-tabpfn_v2_5}"
LANES="${LANES:-4}"                    # one core per lane; matches the scripts' --cpus-per-task=4
MEM="${MEM:-7G}"                       # NHP peaks at ~5.3 GB with 4 lanes (Mila 2026-10-01) x 1.25
TIME="${TIME:-07:00:00}"               # ~1.25 x the slower job (~5 h); jobs requeue and resume from the cell cache
SUBJECTS_A="${SUBJECTS_A:-[1]}"        # 8 channels -> 2 per lane
SUBJECTS_B="${SUBJECTS_B:-[0,3]}"      # 10 channels -> at most 3 per lane
export CONDA_ENV="${CONDA_ENV:-pfns4neurostim}"

# Partition / account come from scripts/cluster.sh when the Narval patch is present; the older scripts carry --partition=main themselves.
if [ -f scripts/cluster.sh ]; then
  read -r -a GPU_FLAGS <<< "$(bash scripts/cluster.sh flags gpu)"
else
  GPU_FLAGS=(--partition=main)
fi

submit() {   # submit <label> <subjects>
  local id
  id=$(LANES="${LANES}" sbatch --parsable ${GPU_FLAGS[@]+"${GPU_FLAGS[@]}"} --mem="${MEM}" --time="${TIME}" --job-name="k2-nhp-$1" scripts/run_stress_sweep.sh "${CONFIG}" "models=[${MODELS}]" "dataset.subjects=$2")
  echo "submitted k2-nhp-$1: subjects $2 -> job ${id%%;*}"
}

submit A "${SUBJECTS_A}"
submit B "${SUBJECTS_B}"
echo "watch with: squeue --me"
