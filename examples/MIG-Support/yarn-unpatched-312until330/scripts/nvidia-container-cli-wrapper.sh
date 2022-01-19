#!/bin/bash


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
                target_gpu_idx="${arg#*=}"
                current_gpu_idx=-1
                while read -r line; do
                    case "$line" in

                        *"<_mig2gpu_device_id>"*)
                            current_gpu_idx=$(($current_gpu_idx+1))
                            if [[ "$current_gpu_idx" == "$target_gpu_idx" && "$line" =~ '<_mig2gpu_device_id>'(.*)'</_mig2gpu_device_id>' ]]; then
                                nvcli_migDeviceId="${BASH_REMATCH[1]}"
                                break
                            fi
                            ;;

                    esac
                done <<< $("$REAL_NVIDIA_SMI_PATH" -q -x | "$THIS_DIR/mig2gpu.sh")

                if [[ "$nvcli_migDeviceId" != "" ]]; then
                    realArgs+=("--device=$nvcli_migDeviceId")
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
