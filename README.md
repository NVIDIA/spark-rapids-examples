# spark-rapids-examples

A repo for Spark related utilities and examples using the Rapids Accelerator,including ETL, ML/DL, etc.

It includes utilities related to running Spark using the Rapids Accelerator, docs and example applications that
demonstrate the RAPIDS.ai GPU-accelerated Spark and Spark-ML(PCA Algorithm) projects.
It now supports Spark 3.0.0+.  It is recommended to set up Spark Cluster with JDK8.

## Utilities and Examples

### 1. Xgboost examples

- Mortgage: [Scala](/examples/mortgage/scala/src/com/nvidia/spark/examples/mortgage), [Python](/examples/mortgage/python/com/nvidia/spark/examples/mortgage)
- Taxi: [Scala](/examples/taxi/scala/src/com/nvidia/spark/examples/taxi), [Python](/examples/taxi/python/com/nvidia/spark/examples/taxi)
- Agaricus: [Scala](/examples/agaricus/scala/src/com/nvidia/spark/examples/agaricus), [Python](/examples/agaricus/python/com/nvidia/spark/examples/agaricus)

### 2. Microbenchmarks

- [Microbenchmarks on CPU](/examples/micro-benchmarks/notebooks/micro-benchmarks-cpu.ipynb)
- [Microbenchmarks on GPU](/examples/micro-benchmarks/notebooks/micro-benchmarks-gpu.ipynb)

### 3. TensorFlow training on Horovod Spark example

- Criteo: [Python](/examples/criteo_train/criteo_keras.py)

### 4. Spark-ML examples
- PCA: [Scala](/examples/pca)

### 5. NVIDIA GPU Plugin for YARN with MIG support
- [YARN MIG GPU Plugin](hadoop/device-plugins/gpu-mig)

## Getting Started Guides

### 1. Xgboost examples guide

Try one of the "Getting Started Guides" below. Please note that they target the Mortgage dataset as written, 
but with a few changes to `EXAMPLE_CLASS` and `dataPath`, they can be easily adapted to the Taxi or Agaricus datasets.

You can get a small size datasets for each example in the [datasets](/datasets) folder. 
These datasets are only provided for convenience. In order to test for performance, 
please prepare a larger dataset by following [Preparing Datasets via Notebook](/datasets/preparing_datasets.md). 
We also provide a larger dataset: [Morgage Dataset (1 GB uncompressed)](https://rapidsai-data.s3.us-east-2.amazonaws.com/spark/mortgage.zip), 
which is used in the guides below.

- Prepare packages and dataset
    - [Scala](/docs/get-started/xgboost-examples/prepare-package-data/preparation-scala.md)
    - [Python](/docs/get-started/xgboost-examples/prepare-package-data/preparation-python.md)
- Getting started on on-premises clusters
    - [Standalone cluster for Scala](/docs/get-started/xgboost-examples/on-prem-cluster/standalone-scala.md)
    - [Standalone cluster for Python](/docs/get-started/xgboost-examples/on-prem-cluster/standalone-python.md)
    - [YARN for Scala](/docs/get-started/xgboost-examples/on-prem-cluster/yarn-scala.md)
    - [YARN for Python](/docs/get-started/xgboost-examples/on-prem-cluster/yarn-python.md)
    - [Kubernetes](/docs/get-started/xgboost-examples/on-prem-cluster/kubernetes.md)
- Getting started on cloud service providers
    - Amazon AWS
        - [EC2](/docs/get-started/xgboost-examples/csp/aws/ec2.md)
    - [Databricks](/docs/get-started/xgboost-examples/csp/databricks/databricks.md)
- Getting started for Jupyter Notebook applications
    - [Apache Toree Notebook for Scala](/docs/get-started/xgboost-examples/notebook/toree.md)
    - [Jupyter Notebook for Python](/docs/get-started/xgboost-examples/notebook/python-notebook.md)

These examples use default parameters for demo purposes. For a full list please see "Supported Parameters" 
for [Scala](/examples/app-parameters/supported_xgboost_parameters_scala.md) 
or [Python](/examples/app-parameters/supported_xgboost_parameters_python.md)

### 2. Microbenchmark guide

The microbenchmark on [RAPIDS Accelerator For Apache Spark](https://nvidia.github.io/spark-rapids/) is to identify, 
test and analyze the best queries which can be accelerated on the GPU. 
The queries are based on several tables in [TPC-DS](http://www.tpc.org/tpcds/) parquet format with Double replacing Decimal,
so that similar speedups can be reproducible by others.
The microbenchmark includes commonly used Spark SQL operations such as expand, hash aggregate, windowing, and cross joins,
and runs the same queries in CPU mode and GPU mode. Some queries will involve data skew.
Each of them is highly [tuned](https://nvidia.github.io/spark-rapids/docs/tuning-guide.html) and works with the optimal configuration
on an 8 nodes Spark standalone cluster which with 128 CPU cores and 1 A100 GPU on each node. 

You can generate the parquet format dataset using this [Databricks Tool](https://github.com/databricks/spark-sql-perf).
All the queries are running on the SF3000(Scale Factors 3TB) dataset. You can generate it with the following command:
```
build/sbt "test:runMain com.databricks.spark.sql.perf.tpcds.GenTPCDSData -d /databricks-tpcds-kit-path -s 3000G -l /your-dataset-path -f parquet"
```
You will see the [RAPIDS Accelerator For Apache Spark](https://nvidia.github.io/spark-rapids/) can give speedups of up to 10x over the CPU, and in some cases up to 80x.
It is easy to compare the [microbenchmarks on CPU](/examples/micro-benchmarks/notebooks/micro-benchmarks-cpu.ipynb) and [GPU](/examples/micro-benchmarks/notebooks/micro-benchmarks-gpu.ipynb) side by side.
You can see some queries are faster in the second time, it can be caused by many reasons such as JVM JIT or initialization overhead or caching input data in the OS page cache, etc.
You can get a clear and visual impression of the improved performance with or without the benefits of post-running.
The improved performance is influenced by many conditions, including the dataset's scale factors or the GPU card.
If the application ran for too long or even failed, you can run the queries on a smaller dataset.

### 3. TensorFlow training on Horovod Spark example guide

Please follow the README guide here: [README](examples/criteo_train/README.md)

### 4. PCA example guide
Please follow the README guide here: [README](/examples/pca/README.md)

## API
### 1. Xgboost examples API

- [Scala API](/docs/api-docs/xgboost-examples-api-docs/scala.md)
- [Python API](/docs/api-docs/xgboost-examples-api-docs/python.md)

## Troubleshooting
- [Trouble Shooting](/docs/trouble-shooting/xgboost-examples-trouble-shooting.md)

## Contributing
See the [Contributing guide](CONTRIBUTING.md).

## Contact Us

Please see the [RAPIDS](https://rapids.ai/community.html) website for contact information.

## License

This content is licensed under the [Apache License 2.0](/LICENSE)
