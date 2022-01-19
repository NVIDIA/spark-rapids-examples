#!/bin/bash

# Copyright (c) 2022, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
