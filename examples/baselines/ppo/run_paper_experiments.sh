#!/bin/bash
# Paper experiments: all tasks, VLM+LLM (failure) + Eureka, with family diversity
# success_once as primary metric (no success_at_end in decisions)

export VLM_CATEGORY_FOCUS_OVERRIDE="failure"
export EXTRA_ARGS_OVERRIDE="--enable_family_diversity"

# GPU assignment: pass as first argument, e.g.:
#   bash run_paper_experiments.sh "0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1"
export GPUS_OVERRIDE="${1:?Usage: bash run_paper_experiments.sh <gpu_list>}"

# All tasks with their TOTAL overrides (use outer_loop_full.sh defaults where possible)
TASKS=(
    "PushCube-v1"
    "PickCube-v1"
    "PushT-v1"
    "AnymalC-Reach-v1"
    "OpenCabinetDoor-v1"
    "OpenCabinetDrawer-v1"
    "PegInsertionSide-v1"
    "UnitreeG1PlaceAppleInBowl-v1"
    "UnitreeG1TransportBox-v1"
    "RotateValveLevel0-v1"
    "RotateSingleObjectInHandLevel0-v1"
)

for ENV in "${TASKS[@]}"; do
    export ENVS_OVERRIDE="${ENV}"
    unset TOTAL_OVERRIDE  # use outer_loop_full.sh per-task defaults

    echo "============================================"
    echo "TASK: ${ENV}"
    echo "============================================"

    # VLM+LLM
    bash outer_loop_full.sh vlm_failureselection

    # Eureka (cross-resume from VLM iter 0, 4 iters)
    CROSS_RESUME_OVERRIDE=1 OUTER_ITERS_OVERRIDE=4 bash outer_loop_full.sh eureka
done

echo "============================================"
echo "ALL PAPER EXPERIMENTS COMPLETE"
echo "============================================"
