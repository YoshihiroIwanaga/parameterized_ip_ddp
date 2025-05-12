#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

export PYTHONPATH=$SCRIPT_DIR/../..:$PYTHONPATH
export OPENBLAS_NUM_THREADS=1
python $SCRIPT_DIR/../../examples/car/run_car_traj_plan_LT20.py
