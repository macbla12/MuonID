#!/bin/bash
# Resolve the directory containing this script (Identification/).
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

export ORT_DIR=$SCRIPT_DIR/onnxruntime
export LD_LIBRARY_PATH=$ORT_DIR/lib:$LD_LIBRARY_PATH
export CPLUS_INCLUDE_PATH=$ORT_DIR/include:$CPLUS_INCLUDE_PATH
export ROOT_INCLUDE_PATH=$ORT_DIR/include:$ROOT_INCLUDE_PATH