#!/bin/bash
# Paper experiments Group 4: GPU 0,1 — remaining heavy tasks
export GPUS_OVERRIDE="0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1"
export VLM_CATEGORY_FOCUS_OVERRIDE="failure"
export EXTRA_ARGS_OVERRIDE="--enable_family_diversity"

for ENV in PegInsertionSide-v1 UnitreeG1TransportBox-v1; do
    export ENVS_OVERRIDE="${ENV}"
    unset TOTAL_OVERRIDE  # use defaults (75M, 100M)

    bash outer_loop_full.sh vlm_failureselection
    CROSS_RESUME_OVERRIDE=1 OUTER_ITERS_OVERRIDE=4 bash outer_loop_full.sh eureka
done
