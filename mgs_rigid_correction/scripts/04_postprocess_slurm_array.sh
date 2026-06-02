#!/bin/bash
# SLURM array script for postprocessing ODB -> CSV extraction on the cluster.
# Each array task processes one ODB file from the flat input directory.

#SBATCH --job-name=mgs_postproc
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=2000
#SBATCH --array=1-180%20
#SBATCH --output=logs/postproc_%A_%a.out
#SBATCH --error=logs/postproc_%A_%a.err

set -e
set -o pipefail

. /etc/profile.d/module.sh
module load abaqus/2023

FLAT_DIR="${SLURM_SUBMIT_DIR}"
METADATA_CSV="${FLAT_DIR}/run_metadata_full.csv"
EXTRACTOR="${FLAT_DIR}/abaqus_extract_odb.py"

if [[ ! -f "${METADATA_CSV}" ]]; then
  echo "ERROR: metadata CSV not found: ${METADATA_CSV}"
  exit 1
fi

TOTAL=$(awk 'NR > 1 {count++} END {print count+0}' "${METADATA_CSV}")
if [[ ${SLURM_ARRAY_TASK_ID} -gt ${TOTAL} ]]; then
  echo "Array index ${SLURM_ARRAY_TASK_ID} > available runs ${TOTAL}; exiting."
  exit 0
fi

LINE_NR=$(( SLURM_ARRAY_TASK_ID + 1 ))

# Extract fields from the metadata CSV using awk.
# Columns: run_id(1), scenario_id(2), D_m(9), S(18), symmetry_factor(38)
read -r run_id scenario_id D_m S symmetry_factor <<EOF
$(awk -F, -v nr="${LINE_NR}" 'NR==nr {
  gsub(/"/, "", $1);  gsub(/"/, "", $2);  gsub(/"/, "", $9);
  gsub(/"/, "", $18); gsub(/"/, "", $38);
  print $1, $2, $9, $18, $38
}' "${METADATA_CSV}")
EOF

if [[ -z "${run_id}" ]]; then
  echo "ERROR: could not read metadata at line ${LINE_NR}"
  exit 2
fi

ODB_PATH="${FLAT_DIR}/${run_id}.odb"
OUTPUT_CSV="${FLAT_DIR}/${run_id}.csv"

if [[ ! -f "${ODB_PATH}" ]]; then
  echo "WARNING: ODB not found: ${ODB_PATH}; skipping."
  exit 0
fi

if [[ -f "${OUTPUT_CSV}" ]]; then
  echo "CSV already exists: ${OUTPUT_CSV}; skipping."
  exit 0
fi

# Write config JSON for abaqus_extract_odb.py
CONFIG="/tmp/postproc_config_${SLURM_JOBID}_${SLURM_ARRAY_TASK_ID}.json"
cat > "${CONFIG}" << JSONEOF
{
  "odb_path": "${ODB_PATH}",
  "step_name": "Einpressen",
  "D_m": ${D_m:-0.6},
  "symmetry_factor": ${symmetry_factor:-4},
  "force_unit_scale_to_kN": 1.0,
  "run_id": "${run_id}",
  "scenario_id": "${scenario_id}",
  "S": ${S:-1},
  "output_csv": "${OUTPUT_CSV}"
}
JSONEOF

echo "Processing ${run_id}: D_m=${D_m}, S=${S}, sym=${symmetry_factor}"
abaqus cae noGUI="${EXTRACTOR}" -- --config "${CONFIG}"

rm -f "${CONFIG}"

if [[ -f "${OUTPUT_CSV}" ]]; then
  lines=$(wc -l < "${OUTPUT_CSV}")
  echo "Done: ${OUTPUT_CSV} (${lines} lines)"
else
  echo "ERROR: CSV was not created: ${OUTPUT_CSV}"
  exit 3
fi
