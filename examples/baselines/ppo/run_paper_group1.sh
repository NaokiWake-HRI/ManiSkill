#!/bin/bash
# Paper experiments Group 1: GPU 2,3 — fast tasks first
export GPUS_OVERRIDE="2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3"
export VLM_CATEGORY_FOCUS_OVERRIDE="failure"
export EXTRA_ARGS_OVERRIDE="--enable_family_diversity"

for ENV in PushCube-v1 PickCube-v1 AnymalC-Reach-v1 PushT-v1 OpenCabinetDrawer-v1 UnitreeG1PlaceAppleInBowl-v1; do
    export ENVS_OVERRIDE="${ENV}"

    # Per-task TOTAL (reduced where validated)
    case "${ENV}" in
        PushCube-v1|PickCube-v1|OpenCabinetDrawer-v1) export TOTAL_OVERRIDE=5_000_000 ;;
        *) unset TOTAL_OVERRIDE ;;  # use outer_loop_full.sh default
    esac

    bash outer_loop_full.sh vlm_failureselection
    CROSS_RESUME_OVERRIDE=1 OUTER_ITERS_OVERRIDE=4 bash outer_loop_full.sh eureka
done
