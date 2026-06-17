#!/usr/bin/env bash
#
# run_kscale_single_dataset.sh — sweep several k_scale_um values per dataset.
#
# For EACH dataset it launches one detached `screen` session running
# kScale_Sweep_HPC.jl, which computes the masks / distance transforms / formation
# times once and then measures curvature at each scale (comparison figures across
# scales). Pass a comma-separated list of datasets to run several at once — one
# screen per dataset, each doing the full k-sweep on that single dataset.
#
# Usage:
#   ./run_kscale_single_dataset.sh                                      # settings below
#   ./run_kscale_single_dataset.sh FM40-1-R1                            # one dataset
#   ./run_kscale_single_dataset.sh FM40-1-R1,FM40-1-R2,FM40-2-R2        # 3 datasets → 3 screens
#   ./run_kscale_single_dataset.sh FM40-1-R1,FM40-1-R2 20,60,100        # datasets + scales (2nd arg)
#   ./run_kscale_single_dataset.sh FM40-1-R1 20,60,100 CTF             # + contour-mean method (3rd arg)
# (datasets and scales are comma-separated lists with NO spaces.)
# mean_method is one of CCF (circle fit, default), CTF (turning fit) or ALF.

set -euo pipefail

# ── Settings ─────────────────────────────────────────────────────────────────
datasets="${1:-FM40-1-R1}"           # comma-separated list, NO spaces — one screen per dataset (1st CLI arg)
k_values="${2:-20,60,100}"           # comma-separated scales [µm], NO spaces (2nd CLI arg overrides)
mean_method="${3:-CCF}"              # contour-mean method: CCF (default), CTF or ALF (3rd CLI arg)
JULIA="julia"                         # julia command (or full path)
PRECOMPILE=true                       # precompile the env once before launching

# If julia comes from a module system, set this (leave empty if it's on PATH):
#   setup_cmds="module load julia/1.10.5"
setup_cmds=""

# ── Paths (derived from this script's location) ──────────────────────────────
HPC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"    # OPALSx/hpc
PROJECT_ROOT="$(dirname "$HPC_DIR")"                        # OPALSx
JL_SCRIPT="$PROJECT_ROOT/scripts/Osteon_Formation_Analysis/kScale_Sweep_HPC.jl"
LOG_DIR="$PROJECT_ROOT/output/logs"
mkdir -p "$LOG_DIR"

[ -n "$setup_cmds" ] && eval "$setup_cmds"

# ── Sanity checks ────────────────────────────────────────────────────────────
command -v screen  >/dev/null || { echo "ERROR: 'screen' not found on PATH."; exit 1; }
command -v "$JULIA" >/dev/null || { echo "ERROR: julia ('$JULIA') not found on PATH."; exit 1; }
[ -f "$JL_SCRIPT" ]            || { echo "ERROR: cannot find $JL_SCRIPT"; exit 1; }

IFS=',' read -ra dataset_list <<< "$datasets"   # split the comma-separated datasets

echo "Project : $PROJECT_ROOT"
echo "Env     : $HPC_DIR"
echo "Datasets: $datasets   (${#dataset_list[@]} screen(s))"
echo "Scales  : $k_values µm"
echo "Mean    : $mean_method"
echo

# ── Precompile once (env must already be instantiated on a login node) ───────
if [ "$PRECOMPILE" = true ]; then
    echo "Precompiling the hpc environment (once) ..."
    "$JULIA" --project="$HPC_DIR" -e 'using Pkg; Pkg.precompile()'
    echo
fi

# ── Launch one detached screen per dataset ───────────────────────────────────
for ds in "${dataset_list[@]}"; do
    ds="${ds// /}"                       # strip any stray spaces
    [ -z "$ds" ] && continue
    run_name="ksweep_${ds}"
    session="opalsx_${run_name}"
    log="$LOG_DIR/${run_name}.log"
    screen -S "$session" -X quit >/dev/null 2>&1 || true   # drop any stale session

    cmd="${setup_cmds:+$setup_cmds; }cd '$PROJECT_ROOT' && '$JULIA' --project='$HPC_DIR' '$JL_SCRIPT' --dataset='$ds' --k='$k_values' --mean_method='$mean_method' --run='$run_name' 2>&1 | tee '$log'"
    screen -dmS "$session" bash -lc "$cmd"
    echo "▶ launched screen '$session'  (dataset=$ds, k=$k_values, mean_method=$mean_method)  → log: $log"
done

echo
echo "Launched ${#dataset_list[@]} run(s). Monitor with:"
echo "  screen -ls                              # list running sessions"
echo "  screen -r opalsx_ksweep_<dataset>       # attach (detach again: Ctrl-a then d)"
echo "  tail -f $LOG_DIR/ksweep_<dataset>.log"
echo
echo "Outputs will be in: $PROJECT_ROOT/output/ksweep_<dataset>/  (and .zip)"
