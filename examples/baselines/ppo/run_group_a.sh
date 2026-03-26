#!/bin/bash
# Group A: GPU 0,1 — 3 tasks sequentially
export GPUS_OVERRIDE="0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1"
export ENVS_OVERRIDE="PegInsertionSide-v1,PushT-v1,UnitreeG1PlaceAppleInBowl-v1"
exec bash outer_loop_full.sh "$@"
