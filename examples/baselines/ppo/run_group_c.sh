#!/bin/bash
# Group C: GPU 4,5 — remaining tasks
export GPUS_OVERRIDE="4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5"
export ENVS_OVERRIDE="PushCube-v1,PickCube-v1,AnymalC-Reach-v1,OpenCabinetDrawer-v1"
exec bash outer_loop_full.sh "$@"
