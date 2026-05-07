#!/bin/bash

HOST="${MGS_CLUSTER_HOST:-hpc4.rz.tuhh.de}"
USER="${MGS_CLUSTER_USER:-cda6556}"
PASSWORD="${MGS_CLUSTER_PASSWORD:-xxxxxxxx}"

sshpass -p "${PASSWORD}" ssh -o StrictHostKeyChecking=no "${USER}@${HOST}"
read -p "Press Enter to exit..."
