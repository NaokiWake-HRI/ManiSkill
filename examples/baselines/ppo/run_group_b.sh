#!/bin/bash
# Group B: GPU 2,3 — PushT + PlaceApple
# Per-task: vlm_failureselection then eureka (cross-resume from vlm iter 0)
export GPUS_OVERRIDE="2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3"

for ENV in PushT-v1 UnitreeG1PlaceAppleInBowl-v1; do
    export ENVS_OVERRIDE="${ENV}"
    bash outer_loop_full.sh vlm_failureselection "$@"
    CROSS_RESUME_OVERRIDE=1 bash outer_loop_full.sh eureka "$@"
done
