#!/bin/bash

set -e


ENABLE_NON_MIG_GPUS=${ENABLE_NON_MIG_GPUS:-0}

# For stored input test: NVIDIA_SMI_QX=./src/resources/tom-nvidia-smi-xq.xml
# For live input test: NVIDIA_SMI_QX=/dev/stdin
NVIDIA_SMI_QX="${NVIDIA_SMI_QX:-"/dev/stdin"}"

mig2gpu_inputLines=()

# buffer global output here
mig2gpu_out=()


mig2gpu_migEnabled=0

mig2gpu_driverVersion="INVALID_DRIVER_VERSION"

# buffer non-MIG GPU output here
mig2gpu_nonMigGpu_out=()
mig2gpu_migGpu_out=()

# Slice of original XML defining the current GPU element
mig2gpu_gpu_lineNumberStart=-1
mig2gpu_gpu_lineNumberEnd=-1

# Slice of original XML defining the current MIG element
mig2gpu_mig_lineNumberStart=-1
mig2gpu_mig_lineNumberEnd=-1
mig2gpu_migIndex=-1

# Parent GPU context for MIG
mig2gpu_gpuIdx=-1
mig2gpu_migGpuInstanceId=-1
mig2gpu_migComputeInstanceUuid=-1
mig2gpu_productName="INVALID_GPU_PRODUCT_NAME"
mig2gpu_gpuUuid="INVALID_GPU_UUID"
mig2gpu_gpuMinorNumber="INVALID_GPU_MINOR_NUMBER"
mig2gpu_gpu_utilization_lineNumberStart=-1
mig2gpu_gpu_utilization_lineNumberEnd=-1
mig2gpu_gpu_temperature_lineNumberStart=-1
mig2gpu_gpu_temperature_lineNumberEnd=-1


# The function to replace a MIG-enabled GPU with the "fake" GPU device elements
# corresponding to MIG devices contained within the given GPU element
#
# The minimum GPU content YARN needs from GPU for parse to succeed:
#
# <nvidia_smi_log>
#         <driver_version>495.29.05</driver_version>
#         <gpu id="00000000:17:00.0">
#                 <product_name>Quadro RTX 6000</product_name>
#                 <uuid>GPU-903720f4-f8d1-11e0-3b2f-4bd740b2f424</uuid>
#                 <minor_number>0</minor_number>
#                 <fb_memory_usage>
#                         <used>673 MiB</used>
#                         <free>&23547 MiB</free>
#                 </fb_memory_usage>
#                 <utilization>
#                         <gpu_util>&23 %</gpu_util>
#                 </utilization>
#                 <temperature>
#                         <gpu_temp>38 C</gpu_temp>
#                         <gpu_temp_max_threshold>94 C</gpu_temp_max_threshold>
#                         <gpu_temp_slow_threshold>91 C</gpu_temp_slow_threshold>
#                 </temperature>
#         </gpu>
# </nvidia_smi_log>
#
# A MIG device looks like this:
# <mig_device>
#     <index>0</index>
#     <gpu_instance_id>3</gpu_instance_id>
#     <compute_instance_id>0</compute_instance_id>
#     <device_attributes>
#         <shared>
#             <multiprocessor_count>14</multiprocessor_count>
#             <copy_engine_count>1</copy_engine_count>
#             <encoder_count>0</encoder_count>
#             <decoder_count>1</decoder_count>
#             <ofa_count>0</ofa_count>
#             <jpg_count>0</jpg_count>
#         </shared>
#     </device_attributes>
#     <ecc_error_count>
#         <volatile_count>
#             <sram_uncorrectable>0</sram_uncorrectable>
#         </volatile_count>
#     </ecc_error_count>
#     <fb_memory_usage>
#         <total>6016 MiB</total>
#         <used>3 MiB</used>
#         <free>6012 MiB</free>
#     </fb_memory_usage>
#     <bar1_memory_usage>
#         <total>8191 MiB</total>
#         <used>0 MiB</used>
#         <free>8191 MiB</free>
#     </bar1_memory_usage>
# </mig_device>
#
# To satisfy the minimum parseable GPU element, we need to
# 1) add a <product_name> element, parent's orginal text + MIG + index
# 2) add a <uuid> element accoring to https://docs.nvidia.com/datacenter/tesla/mig-user-guide/#cuda-gi
#    MIG-<parent gpu uuid>/<gpu instance id>/<compute instance id>
# 3) add parent's <minor_number> 0 (don't care)
# 4) use MIG's own <fb_memory_usage> element unchanged
# 5) copy <utilization> element from parent
# 6) copy <temperature> element from parent
#
# To enable bidirectional translation <mig_device> to/from fake <gpu>
# 7) add a <_mig2gpu_device_id> element: "<parent gpu index>:<mig index>", e.g. 0:0


function processParentGpuGlobals {
    local lineNumber

    # increment 0-based GPU iteration order index
    mig2gpu_gpuIdx=$((mig2gpu_gpuIdx+1))

    for ((lineNumber=mig2gpu_gpu_lineNumberStart; lineNumber<mig2gpu_gpu_lineNumberEnd; lineNumber++)); do
        local line="${mig2gpu_inputLines[$lineNumber]}"

        case "$line" in

            $'\t'*'<current_mig>'*'</current_mig>')
                if [[ "$line" =~ '<current_mig>Enabled</current_mig>' ]]; then
                    mig2gpu_migEnabled=1
                fi
                ;;

            $'\t'*'<product_name>'*)
                if [[ "$line" =~ $'\t\t<product_name>'(.*)'</product_name>' ]]; then
                    mig2gpu_productName="${BASH_REMATCH[1]}"
                fi
                ;;

            $'\t'*'<uuid>'*)
                if [[ "$line" =~ $'\t\t<uuid>'(.*)'</uuid>' ]]; then
                    mig2gpu_gpuUuid="${BASH_REMATCH[1]}"
                fi
                ;;

            $'\t'*'<minor_number>'*)
                mig2gpu_gpuMinorNumber="$line"
                ;;

            $'\t'*'<utilization>'*)
                mig2gpu_gpu_utilization_lineNumberStart="$lineNumber"
                ;;

            $'\t'*'</utilization>'*)
                mig2gpu_gpu_utilization_lineNumberEnd=$((lineNumber+1))
                ;;

            $'\t'*'<temperature>'*)
                mig2gpu_gpu_temperature_lineNumberStart="$lineNumber"
                ;;

            $'\t'*'</temperature>'*)
                mig2gpu_gpu_temperature_lineNumberEnd=$((lineNumber+1))
                ;;
        esac
    done
}


function addOriginalGpuIndexAsDeviceId {
    local afterUuidLineStart=$((mig2gpu_gpu_lineNumberStart+3))
    local afterUuidGpuLength=$((mig2gpu_gpu_lineNumberEnd-afterUuidLineStart))
    mig2gpu_nonMigGpu_out+=( "${mig2gpu_inputLines[@]:$mig2gpu_gpu_lineNumberStart:3}" )
    mig2gpu_nonMigGpu_out+=( $'\t\t'"<_mig2gpu_device_id>$mig2gpu_gpuIdx</_mig2gpu_device_id>")
    mig2gpu_nonMigGpu_out+=( "${mig2gpu_inputLines[@]:$afterUuidLineStart:$afterUuidGpuLength}" )
}


function replaceParentGpuWithMigs {

    for ((lineNumber=mig2gpu_gpu_lineNumberStart; lineNumber<mig2gpu_gpu_lineNumberEnd; lineNumber++)); do
        local line="${mig2gpu_inputLines[$lineNumber]}"

        case "$line" in

            $'\t'*'<mig_device>'*)
                mig2gpu_mig_lineNumberStart=$lineNumber
                ;;

            $'\t'*'<index>'*)
                if [[ "$line" =~ $'\t'*'<index>'(.*)'</index>' ]]; then
                    mig2gpu_migIndex="${BASH_REMATCH[1]}"
                fi
                ;;

            $'\t'*'_instance_id>'*)
                if [[ "$line" =~ $'\t'*'<gpu_instance_id>'(.*)'</gpu_instance_id>' ]]; then
                    mig2gpu_migGpuInstanceId="${BASH_REMATCH[1]}"
                elif [[ "$line" =~ $'\t'*'<compute_instance_id>'(.*)'</compute_instance_id>' ]]; then
                    mig2gpu_migComputeInstanceId="${BASH_REMATCH[1]}"
                fi
                ;;

            $'\t'*'<fb_memory_usage>'*)
                local fbMemoryUsage_lineNumberStart=$lineNumber
                ;;

            $'\t'*'</fb_memory_usage>'*)
                local fbMemoryUsage_lineNumberEnd=$((lineNumber+1))
                local fbMemryUsageLength=$((fbMemoryUsage_lineNumberEnd-fbMemoryUsage_lineNumberStart))
                local fbMemoryUsage=("${mig2gpu_inputLines[@]:$fbMemoryUsage_lineNumberStart:fbMemryUsageLength}")
                local migFbMemoryUsage=("${fbMemoryUsage[@]//$'\t\t\t'/$'\t\t'}")
                ;;

            $'\t'*'</mig_device>'*)
                mig2gpu_mig_lineNumberEnd=$((lineNumber+1))

                # <gpu id="...">
                mig2gpu_migGpu_out+=("${mig2gpu_inputLines[$mig2gpu_gpu_lineNumberStart]}")
                mig2gpu_migGpu_out+=($'\t\t'"<product_name>$mig2gpu_productName (MIG)</product_name>")


                # We don't really use it since driver-dependent
                # but R450 & R460 form is more useful for debugging
                # https://docs.nvidia.com/datacenter/tesla/mig-user-guide/#cuda-visible-devices
                #
                local migUuid="MIG-$mig2gpu_gpuUuid/$mig2gpu_migGpuInstanceId/$mig2gpu_migComputeInstanceId"
                mig2gpu_migGpu_out+=($'\t\t'"<uuid>$migUuid</uuid>")

                # https://github.com/NVIDIA/nvidia-container-runtime#nvidia_visible_devices
                # The scheme <GPU Device Index>:<MIG Device Index> is not annotated with any
                # driver version caveats, so adding this for stability and simplicity
                local migDeviceId="$mig2gpu_gpuIdx:$mig2gpu_migIndex"
                mig2gpu_migGpu_out+=($'\t\t'"<_mig2gpu_device_id>$migDeviceId</_mig2gpu_device_id>")

                mig2gpu_migGpu_out+=("$mig2gpu_gpuMinorNumber")
                mig2gpu_migGpu_out+=("${migFbMemoryUsage[@]}")

                local gpuUtilizationLength=$((mig2gpu_gpu_utilization_lineNumberEnd - mig2gpu_gpu_utilization_lineNumberStart))
                local gpuUtilization=("${mig2gpu_inputLines[@]:$mig2gpu_gpu_utilization_lineNumberStart:gpuUtilizationLength}")
                mig2gpu_migGpu_out+=("${gpuUtilization[@]}")

                local gpuTemperatureLength=$((mig2gpu_gpu_temperature_lineNumberEnd - mig2gpu_gpu_temperature_lineNumberStart))
                mig2gpu_migGpu_out+=("${mig2gpu_inputLines[@]:$mig2gpu_gpu_temperature_lineNumberStart:$gpuTemperatureLength}")

                # </gpu>
                mig2gpu_migGpu_out+=("${mig2gpu_inputLines[$((mig2gpu_gpu_lineNumberEnd-1))]}")
                ;;
        esac
    done
}


function processGpuElement {
    processParentGpuGlobals

    if [[ "$mig2gpu_migEnabled" != "1" ]]; then
        addOriginalGpuIndexAsDeviceId
    else
        # scan gpu element lines twice because the mig section appears before
        # the info needed from parent
        replaceParentGpuWithMigs
    fi
}


function mig2gpuMain {
    local line
    local lineNumber=-1

    # simplified regex-free parser relying on the fact
    # that nvidia-smi output is pretty-printed with tabs
    while IFS= read -r line; do
        lineNumber=$((lineNumber+1))
        mig2gpu_inputLines+=("$line")

        case "$line" in

            # document-level tags
            '<'*)
                mig2gpu_out+=("$line")
                ;;

            $'\t<gpu'*)
                # start of a new GPU element
                mig2gpu_gpu_lineNumberStart="$lineNumber"
                ;;

            $'\t</gpu'*)
                # end of a GPU element
                mig2gpu_gpu_lineNumberEnd=$((lineNumber+1))
                processGpuElement
                ;;

            $'\t<driver_version>'*)
                mig2gpu_driverVersion="$line"
                ;;

            *)
                # ignore infeasible
                ;;

        esac
    done < "$NVIDIA_SMI_QX"

    for outLine in "${mig2gpu_out[@]}"; do
        printf '%s\n' "$outLine"
        if [[ "$outLine" =~ '<nvidia_smi_log>' ]]; then
            printf '%s\n' "$mig2gpu_driverVersion"
            printf '%s\n' "${mig2gpu_migGpu_out[@]}"

            # output non-MIG only if ENABLE_NON_MIG_GPUS is set
            # https://docs.nvidia.com/datacenter/tesla/mig-user-guide/#cuda-visible-devices
            # currently mixing MIG and non-MIG GPUs is not supported by the driver
            # "Note that these constraints may be relaxed in future NVIDIA driver releases for MIG"
            if [[ "${#mig2gpu_migGpu_out[@]}" == "0" || "$ENABLE_NON_MIG_GPUS" == "1" ]]; then
                printf '%s\n' "${mig2gpu_nonMigGpu_out[@]}"
            fi
        fi
    done
}


mig2gpuMain

