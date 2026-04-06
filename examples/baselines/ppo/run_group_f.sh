#!/bin/bash
# Group F: GPU 0,1 — PushT with failure_and_near_miss VLM focus
# Per-task: vlm_failureselection (with near_miss) then eureka (cross-resume, 4 iters)
export GPUS_OVERRIDE="0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1"
export VLM_CATEGORY_FOCUS_OVERRIDE="failure_and_near_miss"

for ENV in PushT-v1; do
    export ENVS_OVERRIDE="${ENV}"
    # bash outer_loop_full.sh vlm_failureselection "$@"
    CROSS_RESUME_OVERRIDE=1 OUTER_ITERS_OVERRIDE=4 bash outer_loop_full.sh eureka "$@"
done
