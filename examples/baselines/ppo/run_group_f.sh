#!/bin/bash
# Group F: GPU 0 — OpenCabinetDoor + OpenCabinetDrawer (8 concurrent pool, total=5M)
export GPUS_OVERRIDE="0,0,0,0,0,0,0,0"
export TOTAL_OVERRIDE=10_000_000
export ENVS_OVERRIDE="OpenCabinetDoor-v1,OpenCabinetDrawer-v1"
exec bash outer_loop_full.sh "$@"
