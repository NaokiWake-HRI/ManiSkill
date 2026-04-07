#!/bin/bash
# Group D: GPU 0 — RotateValve + OpenCabinetDoor + OpenCabinetDrawer
# Per-task: vlm_failureselection then eureka (cross-resume from vlm iter 0)
export GPUS_OVERRIDE="0,0,0,0,0,0,0,0"

for ENV in RotateValveLevel0-v1 OpenCabinetDoor-v1 OpenCabinetDrawer-v1; do
    export ENVS_OVERRIDE="${ENV}"

    # Per-task total_timesteps
    if [ "${ENV}" == "RotateValveLevel0-v1" ]; then
        export TOTAL_OVERRIDE=10_000_000
    else
        export TOTAL_OVERRIDE=5_000_000
    fi

    bash outer_loop_full.sh vlm_failureselection "$@"
    CROSS_RESUME_OVERRIDE=1 bash outer_loop_full.sh eureka "$@"
done
