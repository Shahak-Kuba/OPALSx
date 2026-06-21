#!/usr/bin/env bash
#
# run_k_scale_sweep.sh — launch one detached `screen` session per k_scale_um value,
# each running Multi_Osteon_Analysis_HPC.jl with that scale on the given datasets.
#
# Example: with k_scale_um_array=(20 60 100) it starts three screens running the
# analysis with k_scale_um = 20, 60 and 100 respectively, each writing to its own
# output folder and log.
#
# Usage:
#   ./run_k_scale_sweep.sh                                  # use the settings below
#   ./run_k_scale_sweep.sh FM40-1-R1,FM40-2-R2              # override datasets (1st arg)
#   ./run_k_scale_sweep.sh FM40-1-R1,FM40-2-R2 FM40_        # + run-folder prefix (2nd arg)
#   ./run_k_scale_sweep.sh FM40-4-R2,FM40-4-E2,FM40-2-S2 FM40_ CTF    # + contour-mean method (3rd arg)
# To set a prefix while keeping the default datasets, pass them explicitly:
#   ./run_k_scale_sweep.sh FM40-1-R1,FM40-2-R2 myprefix_
# mean_method is one of CCF (circle fit, default), CTF (turning fit) or ALF
# (average of local fits).
#
# Each value k gets (with <m> = the mean_method):
#   • its own screen session  opalsx_k<k>_<m>
#   • its own output folder    output/k<k>_<m>/  (+ output/k<k>_<m>.zip)
#   • a log file               output/logs/k<k>_<m>.log

set -euo pipefail

# ── Settings ─────────────────────────────────────────────────────────────────
k_scale_um_array=(20 60 100)                   # scales to sweep [µm]
datasets="${1:-FM40-1-R1,FM40-2-R2}"           # comma-separated, NO spaces (1st CLI arg overrides)
run_prefix="${2:-}"                             # output folder name = ${run_prefix}k<value> (2nd CLI arg overrides)
mean_method="${3:-CCF}"                         # contour-mean method: CCF (default), CTF or ALF (3rd CLI arg)
mean_method="${mean_method#:}"; mean_method="$(printf '%s' "$mean_method" | tr '[:lower:]' '[:upper:]')"   # normalise (drop ':', upper-case) so it's consistent in run names
JULIA="julia"                                    # julia command (or full path to the binary)
PRECOMPILE=true                                  # precompile the env once before launching the screens

# If julia is provided through a module system, set this so each screen can find
# it (leave empty if `julia` is already on PATH). Example:
#   setup_cmds="module load julia/1.10.5"
setup_cmds=""

# ── Paths (derived from this script's location) ──────────────────────────────
HPC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"    # OPALSx/hpc
PROJECT_ROOT="$(dirname "$HPC_DIR")"                        # OPALSx
JL_SCRIPT="$PROJECT_ROOT/scripts/Osteon_Formation_Analysis/Multi_Osteon_Analysis_HPC.jl"
LOG_DIR="$PROJECT_ROOT/output/logs"
mkdir -p "$LOG_DIR"

# Make julia available in THIS shell too (for the checks / precompile below).
[ -n "$setup_cmds" ] && eval "$setup_cmds"

# ── Sanity checks ────────────────────────────────────────────────────────────
command -v screen  >/dev/null || { echo "ERROR: 'screen' not found on PATH."; exit 1; }
command -v "$JULIA" >/dev/null || { echo "ERROR: julia ('$JULIA') not found on PATH."; exit 1; }
[ -f "$JL_SCRIPT" ]            || { echo "ERROR: cannot find $JL_SCRIPT"; exit 1; }

echo "Project : $PROJECT_ROOT"
echo "Env     : $HPC_DIR"
echo "Datasets: $datasets"
echo "Scales  : ${k_scale_um_array[*]}"
echo "Mean    : $mean_method"
echo

# ── Precompile once so the parallel screens don't all precompile at the same
#    time (the environment must already be instantiated — do that on a login
#    node with internet: `julia --project=hpc -e 'using Pkg; Pkg.instantiate()'`).
if [ "$PRECOMPILE" = true ]; then
    echo "Precompiling the hpc environment (once) ..."
    "$JULIA" --project="$HPC_DIR" -e 'using Pkg; Pkg.precompile()'
    echo
fi

# ── Launch one detached screen per scale ─────────────────────────────────────
for k in "${k_scale_um_array[@]}"; do
    run="${run_prefix}k${k}_${mean_method}"      # include method so runs don't overwrite
    session="opalsx_${run}"
    log="$LOG_DIR/${run}.log"

    # Drop any stale session of the same name so re-runs start clean.
    screen -S "$session" -X quit >/dev/null 2>&1 || true

    cmd="${setup_cmds:+$setup_cmds; }cd '$PROJECT_ROOT' && '$JULIA' --project='$HPC_DIR' '$JL_SCRIPT' --datasets='$datasets' --k_scale_um=$k --mean_method='$mean_method' --run='$run' 2>&1 | tee '$log'"
    screen -dmS "$session" bash -lc "$cmd"
    echo "▶ launched screen '$session'  (k_scale_um=$k, mean_method=$mean_method)  → log: $log"
done

echo
echo "Launched ${#k_scale_um_array[@]} run(s). Monitor with:"
echo "  screen -ls                       # list running sessions"
echo "  screen -r opalsx_${run_prefix}k<value>_${mean_method}   # attach to one (detach again: Ctrl-a then d)"
echo "  tail -f $LOG_DIR/${run_prefix}k<value>_${mean_method}.log   # follow a run's output"
echo
echo "Outputs will be in: $PROJECT_ROOT/output/<run>/  (and <run>.zip)"
