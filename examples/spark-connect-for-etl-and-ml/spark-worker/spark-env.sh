#!/bin/bash

GPU_COUNT_MAX=$(nvidia-smi -L | wc -l)
export SPARK_WORKER_OPTS="
  -Dspark.worker.resource.gpu.amount=${GPU_COUNT_MAX}
  -Dspark.worker.resource.gpu.discoveryScript=/opt/spark/examples/src/main/scripts/getGpusResources.sh
"