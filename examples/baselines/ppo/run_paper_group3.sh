#!/bin/bash
# Paper experiments Group 3: GPU 4,5 — dexterity + heavy tasks
export GPUS_OVERRIDE="4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5"
export VLM_CATEGORY_FOCUS_OVERRIDE="failure"
export EXTRA_ARGS_OVERRIDE="--enable_family_diversity"

for ENV in RotateValveLevel0-v1 RotateSingleObjectInHandLevel0-v1 UnitreeG1TransportBox-v1; do
    export ENVS_OVERRIDE="${ENV}"

    case "${ENV}" in
        RotateValveLevel0-v1) export TOTAL_OVERRIDE=10_000_000 ;;
        RotateSingleObjectInHandLevel0-v1) ;; # 25M default
        *) unset TOTAL_OVERRIDE ;;
    esac

    bash outer_loop_full.sh vlm_failureselection
    CROSS_RESUME_OVERRIDE=1 OUTER_ITERS_OVERRIDE=4 bash outer_loop_full.sh eureka
done
