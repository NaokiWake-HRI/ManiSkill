#!/bin/bash
# Group B: GPU 2,3 — 3 tasks sequentially
export GPUS_OVERRIDE="2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3"
export ENVS_OVERRIDE="RotateValveLevel0-v1,UnitreeG1TransportBox-v1,OpenCabinetDoor-v1"
exec bash outer_loop_full.sh "$@"
