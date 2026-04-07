#!/bin/bash
# Group D: GPU 2,3,4,5 — RotateSingleObjectInHand with failure_and_near_miss VLM focus
export GPUS_OVERRIDE="2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5"
export VLM_CATEGORY_FOCUS_OVERRIDE="failure_and_near_miss"

for ENV in RotateSingleObjectInHandLevel0-v1; do
    export ENVS_OVERRIDE="${ENV}"

    # Per-task total_timesteps
    # Use outer_loop_full.sh defaults (25M for RotateSingleObject)
    unset TOTAL_OVERRIDE

    bash outer_loop_full.sh vlm_failureselection "$@"
    CROSS_RESUME_OVERRIDE=1 OUTER_ITERS_OVERRIDE=4 bash outer_loop_full.sh eureka "$@"
done
