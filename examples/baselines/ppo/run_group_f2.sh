#!/bin/bash
# Group F2: PushT failure_and_near_miss, resume from failure mode's iter 0
# Ensures same starting point as failure mode for fair comparison
export GPUS_OVERRIDE="0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1"
export VLM_CATEGORY_FOCUS_OVERRIDE="failure_and_near_miss"
export ENVS_OVERRIDE="PushT-v1"
export RESUME_DIR_OVERRIDE="runs/outer-loop_full_failureselection_k_16/PushT-v1/ppo-vlm-full-failureselection-PushT-v1-9351-PushT-v1-20260401_093535"

# VLM+LLM F+NM, resuming from failure mode iter 0
bash outer_loop_full.sh vlm_failureselection

# Eureka F+NM, cross-resume from the VLM run above
CROSS_RESUME_OVERRIDE=1 OUTER_ITERS_OVERRIDE=4 RESUME_DIR_OVERRIDE="" bash outer_loop_full.sh eureka
