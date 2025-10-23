#!/bin/bash

# Copyright (c) 2022-2025, NVIDIA CORPORATION.
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

# This script is executed by the `nvidia` Docker runtime on the host before creating the container.
# It intercepts the device assigned by YARN, a 0-based index and converts it to a pair
# GPU device index:MIG device index that is stored in _mig2gpu_device_is elememnt
# by mig2gpu.sh in the nvidia-smi-wrapper.sh which limits all the processes withing the container
# to the corresponding MIG Compute Instance https://github.com/NVIDIA/nvidia-container-runtime#nvidia_visible_devices.

# customize in /etc/nvidia-container-runtime/config.toml
# [nvidia-container-cli]
# environment = [ "VAR1=VAL1", "VAR2=VAL2" ]
REAL_NVIDIA_CONTAINER_CLI_PATH=${REAL_NVIDIA_CONTAINER_CLI_PATH:-"/usr/bin/nvidia-container-cli"}
REAL_NVIDIA_SMI_PATH=${REAL_NVIDIA_SMI_PATH:-"/usr/bin/nvidia-smi"}
MIG_AS_GPU_ENABLED=${MIG_AS_GPU_ENABLED:-"0"}

THIS_PATH="$(readlink -f $0)"
THIS_DIR="$(dirname $THIS_PATH)"

if [[ "$MIG_AS_GPU_ENABLED" == "1" ]]; then
    realArgs=()
    for arg in "$@"; do
        case "$arg" in

            "--device="*)
                nvcli_migDeviceIds=()
                # map CSV of indexes 0,3,10 to ,0,3,10,
                # so we can do an easy "contains" test
                # the device N is included if deviceArgWithLeadingTrailingComma
                # matches =~ ",N,"
                deviceArgWithLeadingTrailingComma=",${arg#*=},"
                current_gpu_idx=-1
                while read -r line; do
                    case "$line" in

                        # found the device id constructed in mig2gpu.sh with the original nvidia-smi enumeration
                        # gpu index, mig index
                        *"<_mig2gpu_device_id>"*)
                            current_gpu_idx=$((current_gpu_idx+1))
                            if [[ "$deviceArgWithLeadingTrailingComma" =~ ",${current_gpu_idx}," && "$line" =~ '<_mig2gpu_device_id>'(.*)'</_mig2gpu_device_id>' ]]; then
                                nvcli_migDeviceIds+=("${BASH_REMATCH[1]}")
                            fi
                            ;;

                    esac
                done < <("$REAL_NVIDIA_SMI_PATH" -q -x | "$THIS_DIR/mig2gpu.sh")
                # make sure the above redirect into the while read loop does not use the here-string (<<<) method because different
                # versions of bash materialize newlines differently in the string. Older versions treat it as a single
                # line and newer versions leave it as a multiline string. Here it needs to be a multiline.

                if (( ${#nvcli_migDeviceIds[@]} )); then
                    migDeviceIdsCsv=$(IFS=','; echo "${nvcli_migDeviceIds[*]}")
                    realArgs+=("--device=$migDeviceIdsCsv")
                else
                    realArgs+=("$arg")
                fi

                ;;

            *)
                realArgs+=("$arg")
                ;;

        esac
    done
    "$REAL_NVIDIA_CONTAINER_CLI_PATH" "${realArgs[@]}"
else
    "$REAL_NVIDIA_CONTAINER_CLI_PATH" "$@"
fi
