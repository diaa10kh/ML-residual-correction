#!/bin/bash

USER="${MGS_CLUSTER_USER:-cda6556}"
HOSTS="${MGS_CLUSTER_HOSTS:-${MGS_CLUSTER_HOST:-hpc3.rz.tuhh.de hpc2.rz.tuhh.de}}"
SSH_OPTS="${MGS_SSH_OPTS:--o ConnectTimeout=15}"
WORK_DIR="${MGS_ROOT_REMOTE:-/work/gbt/${USER}/ML residual correction Phase0}"

for HOST in ${HOSTS}; do
  echo "Trying ${HOST}..."
  ssh ${SSH_OPTS} -t "${USER}@${HOST}" "cd '${WORK_DIR}' && exec bash"
  rc=$?
  if [[ ${rc} -ne 255 ]]; then
    exit ${rc}
  fi
done

echo "No reachable cluster host found in: ${HOSTS}"
read -p "Press Enter to exit..."
