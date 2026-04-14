#!/bin/bash
# Ablation: OpenCabinetDrawer with failure_and_near_miss VLM focus
# Same as the paper setting (famdiv ON, K=16, success_once-based elite),
# but change only the VLM episode curation from failure-only to
# failure+near_miss. Resume from the paper run's iteration 0 so the
# comparison isolates the effect of what the VLM sees.

export GPUS_OVERRIDE="2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3"
export VLM_CATEGORY_FOCUS_OVERRIDE="failure_and_near_miss"
export EXTRA_ARGS_OVERRIDE="--enable_family_diversity"
export ENVS_OVERRIDE="OpenCabinetDrawer-v1"
export TOTAL_OVERRIDE=10_000_000
export OUTER_ITERS_OVERRIDE=4
export RESUME_DIR_OVERRIDE="runs/outer-loop_full_failureselection_famdiv_k_16/OpenCabinetDrawer-v1/ppo-vlm-full-failureselection-OpenCabinetDrawer-v1-9351-OpenCabinetDrawer-v1-20260410_172709"

bash outer_loop_full.sh vlm_failureselection
