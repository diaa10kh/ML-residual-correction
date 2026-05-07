#!/bin/bash

set -e

# Local WSL helper modeled after the working cluster scripts.
# It uploads generated .inp files into one flat remote directory, copies the
# SLURM submitter and user routine, submits an array, and can fetch result files.
#
# Usage from mgs_rigid_correction:
#   ./scripts/12_cluster_transfer_submit_fetch.sh submit_all 0 0
#   ./scripts/12_cluster_transfer_submit_fetch.sh fetch 0 0

HOST="${MGS_CLUSTER_HOST:-hpc4.rz.tuhh.de}"
USER="${MGS_CLUSTER_USER:-cda6556}"

ROOT_LOCAL="${MGS_ROOT_LOCAL:-.}"
ROOT_REMOTE="${MGS_ROOT_REMOTE:-/work/gbt/${USER}/MGS_Rigid_Correction/PHASE1_FLAT}"
MATRIX_CSV="${MGS_MATRIX_CSV:-data/extracted/run_metadata.csv}"
SLURM_SCRIPT_LOCAL="${MGS_SLURM_SCRIPT:-scripts/03_submit_slurm_array.sh}"
USERRTN_LOCAL="${MGS_USERRTN:-../vumat-hypo-2020-hst.for}"
CLEAR_REMOTE_INPUTS="${MGS_CLEAR_REMOTE_INPUTS:-1}"

MODE="${1:-submit_all}"
START_INDEX="${2:-0}"
END_INDEX="${3:-0}"

ssh_remote() {
  ssh "${USER}@${HOST}" "$1"
}

scp_to_remote() {
  scp "$1" "${USER}@${HOST}:$2"
}

scp_from_remote() {
  scp "${USER}@${HOST}:$1" "$2"
}

metadata_row_count() {
  awk 'NR > 1 {count++} END {print count+0}' "${MATRIX_CSV}"
}

run_id_for_index() {
  local index="$1"
  awk -F, -v row="${index}" 'NR==row+1 {print $1}' "${MATRIX_CSV}"
}

submit_all() {
  local total
  total=$(metadata_row_count)
  if [[ "${total}" -eq 0 ]]; then
    echo "No runs found in ${MATRIX_CSV}"
    exit 0
  fi

  local start="${START_INDEX}"
  local end="${END_INDEX}"
  if [[ "${start}" -le 0 ]]; then
    start=1
  fi
  if [[ "${end}" -le 0 || "${end}" -gt "${total}" ]]; then
    end="${total}"
  fi
  if [[ "${start}" -gt "${end}" ]]; then
    echo "Invalid range ${START_INDEX}-${END_INDEX}"
    exit 2
  fi

  echo "Preparing remote flat directory ${ROOT_REMOTE}"
  ssh_remote "mkdir -p \"${ROOT_REMOTE}/logs\""
  if [[ "${CLEAR_REMOTE_INPUTS}" == "1" ]]; then
    ssh_remote "rm -f \"${ROOT_REMOTE}\"/*.inp \"${ROOT_REMOTE}\"/*.odb \"${ROOT_REMOTE}\"/*.sta \"${ROOT_REMOTE}\"/*.msg \"${ROOT_REMOTE}\"/*.dat \"${ROOT_REMOTE}\"/*.log \"${ROOT_REMOTE}\"/*_output.out"
  fi
  scp_to_remote "${SLURM_SCRIPT_LOCAL}" "${ROOT_REMOTE}/03_submit_slurm_array.sh"
  if [[ -f "${USERRTN_LOCAL}" ]]; then
    scp_to_remote "${USERRTN_LOCAL}" "${ROOT_REMOTE}/vumat-hypo-2020-hst.for"
  else
    echo "WARNING: user subroutine not found locally: ${USERRTN_LOCAL}"
  fi

  local copied=0
  local i
  for ((i=start; i<=end; i++)); do
    local run_id
    run_id=$(run_id_for_index "${i}")
    if [[ -z "${run_id}" ]]; then
      continue
    fi
    local local_inp="${ROOT_LOCAL}/runs/${run_id}/${run_id}.inp"
    if [[ ! -f "${local_inp}" ]]; then
      echo "WARNING: missing .inp, skipping ${run_id}: ${local_inp}"
      continue
    fi
    scp_to_remote "${local_inp}" "${ROOT_REMOTE}/${run_id}.inp"
    copied=$((copied+1))
  done

  if [[ "${copied}" -eq 0 ]]; then
    echo "No .inp files copied. Nothing to submit."
    exit 0
  fi

  echo "Submitting ${copied} jobs with array 1-${copied}%40"
  ssh_remote "cd \"${ROOT_REMOTE}\"; sbatch --array=1-${copied}%40 03_submit_slurm_array.sh"
}

fetch_results() {
  local total
  total=$(metadata_row_count)
  local start="${START_INDEX}"
  local end="${END_INDEX}"
  if [[ "${start}" -le 0 ]]; then
    start=1
  fi
  if [[ "${end}" -le 0 || "${end}" -gt "${total}" ]]; then
    end="${total}"
  fi

  local i
  for ((i=start; i<=end; i++)); do
    local run_id
    run_id=$(run_id_for_index "${i}")
    if [[ -z "${run_id}" ]]; then
      continue
    fi
    mkdir -p "${ROOT_LOCAL}/runs/${run_id}"
    for ext in odb sta msg dat log; do
      scp_from_remote "${ROOT_REMOTE}/${run_id}.${ext}" "${ROOT_LOCAL}/runs/${run_id}/" 2>/dev/null || true
    done
    scp_from_remote "${ROOT_REMOTE}/${run_id}_output.out" "${ROOT_LOCAL}/runs/${run_id}/" 2>/dev/null || true
  done
  echo "Fetch complete for range ${start}-${end}."
}

if [[ "${MODE}" == "submit_all" || "${MODE}" == "submit" ]]; then
  submit_all
elif [[ "${MODE}" == "fetch" ]]; then
  fetch_results
else
  echo "Unknown mode: ${MODE}. Use submit_all or fetch."
  exit 2
fi
