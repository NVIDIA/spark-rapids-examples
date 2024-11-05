sudo rm -f /databricks/jars/spark--maven-trees--ml--10.x--xgboost-gpu--ml.dmlc--xgboost4j-gpu_2.12--ml.dmlc__xgboost4j-gpu_2.12__1.5.2.jar
sudo rm -f /databricks/jars/spark--maven-trees--ml--10.x--xgboost-gpu--ml.dmlc--xgboost4j-spark-gpu_2.12--ml.dmlc__xgboost4j-spark-gpu_2.12__1.5.2.jar

sudo wget -O /databricks/jars/rapids-4-spark_2.12-24.10.0.jar https://repo1.maven.org/maven2/com/nvidia/rapids-4-spark_2.12/24.10.0/rapids-4-spark_2.12-24.10.0.jar
sudo wget -O /databricks/jars/xgboost4j-gpu_2.12-1.7.1.jar https://repo1.maven.org/maven2/ml/dmlc/xgboost4j-gpu_2.12/1.7.1/xgboost4j-gpu_2.12-1.7.1.jar
sudo wget -O /databricks/jars/xgboost4j-spark-gpu_2.12-1.7.1.jar https://repo1.maven.org/maven2/ml/dmlc/xgboost4j-spark-gpu_2.12/1.7.1/xgboost4j-spark-gpu_2.12-1.7.1.jar
ls -ltr

mkdir -p /dbfs/FileStore/tables/
cd /dbfs/FileStore/tables/
# Note that this is just a dummy dataset for quickly hands on, please refer the instructions to download the full dataset:
# https://github.com/NVIDIA/spark-rapids-examples/blob/main/docs/get-started/xgboost-examples/dataset/mortgage.md
wget -O mortgage.zip https://rapidsai-data.s3.us-east-2.amazonaws.com/spark/mortgage.zip
ls
unzip -o mortgage.zip
pwd
ls -ltr mortgage/csv/*