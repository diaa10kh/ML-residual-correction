@echo off

REM Run from mgs_rigid_correction on Windows.
REM Optional environment variables:
REM   MGS_CLUSTER_PASSWORD
REM   MGS_ROOT_REMOTE

wsl ./scripts/12_cluster_transfer_submit_fetch.sh submit_all 0 0
wsl ./scripts/13_cluster_open_terminal.sh
