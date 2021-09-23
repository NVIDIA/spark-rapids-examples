SPARK_HOME=/opt/spark-3.1.2-bin-hadoop3.2
SPARK_URL=spark://127.0.0.1:7077

$SPARK_HOME/bin/spark-submit --master $SPARK_URL --deploy-mode client \
--driver-memory 20G \
--executor-memory 50G \
--executor-cores 6 \
--conf spark.cores.max=96 \
--conf spark.task.cpus=6 \
--conf spark.locality.wait=0 \
--conf spark.yarn.maxAppAttempts=1 \
--conf spark.sql.shuffle.partitions=4 \
--conf spark.sql.files.maxPartitionBytes=1024m \
--conf spark.sql.warehouse.dir=$OUT \
--conf spark.task.resource.gpu.amount=0.08 \
--conf spark.executor.resource.gpu.amount=1 \
--conf spark.executor.resource.gpu.discoveryScript=./getGpusResources.sh \
--files $SPARK_HOME/examples/src/main/scripts/getGpusResources.sh \
./criteo_keras.py \
--num-proc 16 \
--data-dir file:///data/parquet \
--logs-dir $PWD/tf_logs \
--dataloader nvtabular \
--learning-rate 0.001 \
--batch-size 65536 \
--epochs 1 \
--local-checkpoint-file ckpt_file