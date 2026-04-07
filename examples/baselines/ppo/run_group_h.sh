#!/bin/bash
# Group H: GPU 4,5 — RotateValveLevel0 with failure_and_near_miss VLM focus
export GPUS_OVERRIDE="4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5"
export VLM_CATEGORY_FOCUS_OVERRIDE="failure_and_near_miss"
export TOTAL_OVERRIDE=10_000_000

for ENV in RotateValveLevel0-v1; do
    export ENVS_OVERRIDE="${ENV}"
    bash outer_loop_full.sh vlm_failureselection "$@"
    CROSS_RESUME_OVERRIDE=1 OUTER_ITERS_OVERRIDE=4 bash outer_loop_full.sh eureka "$@"
done
