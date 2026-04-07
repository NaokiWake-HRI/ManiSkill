#!/bin/bash
# Group G: GPU 2,3 — OpenCabinetDrawer + OpenCabinetDoor with failure_and_near_miss VLM focus
export GPUS_OVERRIDE="2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3"
export VLM_CATEGORY_FOCUS_OVERRIDE="failure_and_near_miss"
export TOTAL_OVERRIDE=10_000_000

for ENV in OpenCabinetDoor-v1; do
    export ENVS_OVERRIDE="${ENV}"
    bash outer_loop_full.sh vlm_failureselection "$@"
    CROSS_RESUME_OVERRIDE=1 OUTER_ITERS_OVERRIDE=4 bash outer_loop_full.sh eureka "$@"
done
