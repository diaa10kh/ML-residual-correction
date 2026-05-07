#!/bin/bash

HOST="${MGS_CLUSTER_HOST:-hpc4.rz.tuhh.de}"
USER="${MGS_CLUSTER_USER:-cda6556}"
WORK_DIR="${MGS_ROOT_REMOTE:-/work/gbt/${USER}/ML residual correction Phase0}"

ssh -t "${USER}@${HOST}" "cd '${WORK_DIR}' && exec bash"
read -p "Press Enter to exit..."
