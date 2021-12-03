# spark-rapids-examples

A repo for Spark related utilities and examples using the Rapids Accelerator,including ETL, ML/DL, etc.

It includes utilities related to running Spark using the Rapids Accelerator, docs and example applications that
demonstrate the RAPIDS.ai GPU-accelerated Spark and Spark-ML(PCA Algorithm) projects.
Please see the [Rapids Accelerator for Spark documentation](https://nvidia.github.io/spark-rapids/Getting-Started/) for supported
Spark versions and requirements. It is recommended to set up Spark Cluster with JDK8.

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

### 5. YARN 3.3.0+ MIG support
- [YARN 3.3.0+ MIG GPU Plugin](hadoop/device-plugins/gpu-mig)

### 6. YARN 3.1.2 until YARN 3.3.0 MIG support
- [YARN 3.1.2 until YARN 3.3.0 MIG GPU Support](hadoop/resource-types/gpu-mig)

## Getting Started Guides

### 1. Xgboost examples guide

We provide three similar Xgboost benchmarks, Mortgage, Taxi and Agaricus. Try one of the "Getting Started Guides" below. 
Please note that they target the Mortgage dataset as written with a few changes 
to `EXAMPLE_CLASS` and `dataPath`, they can be easily adapted with each other with different datasets.

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
test and analyze the best queries which can be accelerated on the GPU. For detail information please refer to this
[guide](/examples/micro-benchmarks/README.md).

### 3. TensorFlow training on Horovod Spark example guide

We provide a Criteo Benchmark to demo ETL and deep learning training on Horovod Spark, please refer to 
this [guide](examples/criteo_train/README.md).

### 4. PCA example guide

This is an example of the GPU accelerated PCA algorithm running on Spark. For detail information please refer to this
[guide](/examples/pca/README.md).

## API
### 1. Xgboost examples API

These guides focus on GPU related Scala and python API interfaces.
- [Scala API](/docs/api-docs/xgboost-examples-api-docs/scala.md)
- [Python API](/docs/api-docs/xgboost-examples-api-docs/python.md)

## Troubleshooting
You can trouble-shooting issues according to following guides.
- [Trouble Shooting XGBoost](/docs/trouble-shooting/xgboost-examples-trouble-shooting.md)

## Contributing
See the [Contributing guide](CONTRIBUTING.md).

## Contact Us

Please see the [RAPIDS](https://rapids.ai/community.html) website for contact information.

## License

This content is licensed under the [Apache License 2.0](/LICENSE)
