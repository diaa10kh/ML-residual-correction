#!/bin/bash
# Remote SLURM array script for rigid-pile MGS Abaqus jobs.
# This follows the working TUHH cluster pattern from 2D_PullOut_Submit.sh:
# submit from a flat directory of .inp files, copy one selected input to a
# job-specific /work folder, run Abaqus there, then copy result files back.

#SBATCH --job-name=mgs_rigid
#SBATCH --ntasks=1
#SBATCH --time=50:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=5000
#SBATCH --array=1-999%40
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err

set -e

#################################
# -  Settings                   - #
#################################

kuerzel=${MGS_CLUSTER_USER:-cda6556}
institut=${MGS_CLUSTER_INSTITUTE:-gbt}
cpus=${SLURM_CPUS_PER_TASK:-8}
userroutine=${MGS_USERROUTINE:-vumat-hypo-2020-hst.for}
project_name=${MGS_PROJECT_NAME:-MGS_Rigid_Correction}

############################################################
# -  No changes needed beyond this point  -                #
############################################################

echo "Printing initial status information: $(scontrol show job ${SLURM_JOBID})"

. /etc/profile.d/module.sh
module load abaqus/2023
module load intel/2019

input_dir="${SLURM_SUBMIT_DIR}"
mapfile -t input_files < <(find "${input_dir}" -maxdepth 1 -type f -name '*.inp' -printf '%p\n' | sort -V)
num_inputs=${#input_files[@]}

if [[ ${num_inputs} -eq 0 ]]; then
  echo "No .inp files found in ${input_dir}; nothing to run."
  exit 0
fi

if [[ ${SLURM_ARRAY_TASK_ID} -gt ${num_inputs} ]]; then
  echo "Array index ${SLURM_ARRAY_TASK_ID} > available inputs ${num_inputs}; nothing to run. Exiting."
  exit 0
fi

input_file=${input_files[$SLURM_ARRAY_TASK_ID-1]}
jobname=$(basename "${input_file}" .inp)

fullpath="/work/${institut}/${kuerzel}/${project_name}/${SLURM_JOBID}_${jobname}"
mkdir -p "${fullpath}"

cp "${input_file}" "${fullpath}/"
if [[ -f "${input_dir}/${userroutine}" ]]; then
  cp "${input_dir}/${userroutine}" "${fullpath}/"
fi

cd "${fullpath}"
if [[ -f "${userroutine}" ]]; then
  abaqus job="${jobname}" input="${jobname}.inp" user="${userroutine}" cpus="${cpus}" interactive double | tee "${jobname}_output.out"
else
  echo "WARNING: user routine ${userroutine} not found; running without user=..."
  abaqus job="${jobname}" input="${jobname}.inp" cpus="${cpus}" interactive double | tee "${jobname}_output.out"
fi

for ext in odb sta msg dat log; do
  if [[ -f "${jobname}.${ext}" ]]; then
    cp "${jobname}.${ext}" "${SLURM_SUBMIT_DIR}/"
  fi
done
if [[ -f "${jobname}_output.out" ]]; then
  cp "${jobname}_output.out" "${SLURM_SUBMIT_DIR}/"
fi
