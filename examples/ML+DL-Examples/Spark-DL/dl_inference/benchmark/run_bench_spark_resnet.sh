#!/bin/bash

# Help function
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo "Run Spark ResNet predict_batch_udf benchmark with different configurations"
    echo ""
    echo "Options:"
    echo "  -t, --type TYPE               Benchmark type: base or triton"
    echo "  -b, --batch-size SIZE         Batch size used in each task of predict_batch_udf"
    echo ""
    echo "Examples:"
    echo "  $0 -t triton -b 256"
    echo "  $0 -t base -b 1024"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--type)
            BENCH_TYPE="$2"
            shift 2
            ;;
        -b|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

TASK_CPUS=1
TASK_GPU_AMOUNT=0.0625
if [ "$BENCH_TYPE" = "base" ]; then
    TASK_GPU_AMOUNT=1
    TASK_CPUS=16
fi

echo "Running benchmark with configuration:"
echo "  Benchmark type: $BENCH_TYPE"
echo "  Task GPU amount: $TASK_GPU_AMOUNT"
echo "  Task CPUs: $TASK_CPUS"

if [ -n "$BATCH_SIZE" ]; then
    echo "  Batch size: $BATCH_SIZE"
    BATCH_SIZE_ARG="--batch-size $BATCH_SIZE"
else
    if [ "$BENCH_TYPE" = "triton" ]; then
        echo "  Batch size: 256 (default for triton)"
        BATCH_SIZE_ARG="--batch-size 256"
    else
        echo "  Batch size: 1024 (default for base)"
        BATCH_SIZE_ARG="--batch-size 1024"
    fi
fi

if [ "$BENCH_TYPE" = "triton" ]; then
    SCRIPT_ARGS="$BATCH_SIZE_ARG --use-triton"
else
    SCRIPT_ARGS="$BATCH_SIZE_ARG"
fi

spark-submit \
    --master spark://$(hostname):7077 \
    --num-executors 1 \
    --executor-cores 16 \
    --executor-memory 32g \
    --conf spark.executor.resource.gpu.amount=1 \
    --conf spark.task.resource.gpu.amount=$TASK_GPU_AMOUNT \
    --conf spark.task.cpus=$TASK_CPUS \
    --conf spark.task.maxFailures=1 \
    --conf spark.sql.execution.arrow.pyspark.enabled=true \
    --conf spark.python.worker.reuse=true \
    --conf spark.pyspark.python=${CONDA_PREFIX}/bin/python \
    --conf spark.pyspark.driver.python=${CONDA_PREFIX}/bin/python \
    --conf spark.locality.wait=0s \
    --conf spark.sql.adaptive.enabled=false \
    --conf spark.sql.execution.sortBeforeRepartition=false \
    --conf spark.sql.files.minPartitionNum=16 \
    bench_spark_resnet.py $SCRIPT_ARGS
