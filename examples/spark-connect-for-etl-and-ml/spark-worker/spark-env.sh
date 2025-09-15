#!/bin/bash

GPU_COUNT_MAX=$(nvidia-smi -L | wc -l)
export SPARK_WORKER_OPTS="
  -Dspark.worker.resource.gpu.amount=${GPU_COUNT_MAX}
  -Dspark.worker.resource.gpu.discoveryScript=/opt/spark/examples/src/main/scripts/getGpusResources.sh
"

# workaround for wheels installation not setting the correct LD_LIBRARY_PATH
#https://github.com/rapidsai/cuml/issues/5300#issuecomment-2084646729
export LD_LIBRARY_PATH=$(find /usr/local/lib/python3.10/dist-packages/nvidia -name lib -type d | xargs printf '%s:'):$LD_LIBRARY_PATH
