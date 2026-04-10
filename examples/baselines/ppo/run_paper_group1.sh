#!/bin/bash
# Paper experiments Group 1: GPU 0,1 — fast tasks
export GPUS_OVERRIDE="0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1"
export VLM_CATEGORY_FOCUS_OVERRIDE="failure"
export EXTRA_ARGS_OVERRIDE="--enable_family_diversity"

for ENV in PushCube-v1 PickCube-v1 AnymalC-Reach-v1 PushT-v1; do
    export ENVS_OVERRIDE="${ENV}"

    case "${ENV}" in
        PushCube-v1|PickCube-v1) export TOTAL_OVERRIDE=5_000_000 ;;
        *) unset TOTAL_OVERRIDE ;;
    esac

    bash outer_loop_full.sh vlm_failureselection
    CROSS_RESUME_OVERRIDE=1 OUTER_ITERS_OVERRIDE=4 bash outer_loop_full.sh eureka
done
