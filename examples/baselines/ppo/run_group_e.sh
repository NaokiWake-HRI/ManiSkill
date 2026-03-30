#!/bin/bash
# Group E: Eureka (LLM-only) K=16 — resume iter 0 from VLM+LLM FailSel K=16 counterpart
export GPUS_OVERRIDE="0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1"
export ENVS_OVERRIDE="PushCube-v1,AnymalC-Reach-v1,PushT-v1,UnitreeG1PlaceAppleInBowl-v1"
export CROSS_RESUME_OVERRIDE=1
exec bash outer_loop_full.sh eureka
