#!/bin/bash

# customize in yarn-env.sh
REAL_NVIDIA_SMI_PATH=${REAL_NVIDIA_SMI_PATH:-"/usr/bin/nvidia-smi"}
MIG_AS_GPU_ENABLED=${MIG_AS_GPU_ENABLED:-"0"}

THIS_PATH="$(readlink -f $0)"
THIS_DIR="$(dirname $THIS_PATH)"

for arg in "$@"; do
    case "$arg" in

        "-q"|"--query")
            QUERY_ARG=1
            ;;

        "-x"|"--xml-format")
            XML_FORMAT_ARG=1
            ;;

    esac
done

if [[ "$MIG_AS_GPU_ENABLED" == "1" && "$XML_FORMAT_ARG" == "1" && "$QUERY_ARG" == "1" ]]; then
    "$REAL_NVIDIA_SMI_PATH" "$@" | "$THIS_DIR/mig2gpu.sh"
else
    "$REAL_NVIDIA_SMI_PATH" "$@"
fi
