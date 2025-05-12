#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

export PYTHONPATH=$SCRIPT_DIR/../..:$PYTHONPATH
export OPENBLAS_NUM_THREADS=1

python $SCRIPT_DIR/../../examples/inverted_pendulum/find_inverted_pendulum_margin_and_cp_zero_order.py
python $SCRIPT_DIR/../../examples/inverted_pendulum/find_inverted_pendulum_margin_and_cp_linear.py
python $SCRIPT_DIR/../../examples/inverted_pendulum/find_inverted_pendulum_margin_and_cp_cubic.py

