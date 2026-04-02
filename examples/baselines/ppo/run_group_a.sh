#!/bin/bash
# Group A: GPU 0,1 — PushCube + AnymalC
# Per-task: vlm_failureselection then eureka (cross-resume from vlm iter 0)
export GPUS_OVERRIDE="0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1"

for ENV in PushCube-v1 AnymalC-Reach-v1; do
    export ENVS_OVERRIDE="${ENV}"
    bash outer_loop_full.sh vlm_failureselection "$@"
    CROSS_RESUME_OVERRIDE=1 bash outer_loop_full.sh eureka "$@"
done
