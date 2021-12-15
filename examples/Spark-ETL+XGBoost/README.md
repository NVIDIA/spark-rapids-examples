# Spark-ETL+XGBoost examples 

## 1. Spark-ETL+XGBoost examples codes

- Mortgage: [Scala](../../examples/Spark-ETL+XGBoost/mortgage/scala/src/com/nvidia/spark/examples/mortgage), [Python](../../examples/Spark-ETL+XGBoost/mortgage/python/com/nvidia/spark/examples/mortgage)
- Taxi: [Scala](../../examples/Spark-ETL+XGBoost/taxi/scala/src/com/nvidia/spark/examples/taxi), [Python](../../examples/Spark-ETL+XGBoost/taxi/python/com/nvidia/spark/examples/taxi)
- Agaricus: [Scala](../../examples/Spark-ETL+XGBoost/agaricus/scala/src/com/nvidia/spark/examples/agaricus), [Python](../../examples/Spark-ETL+XGBoost/agaricus/python/com/nvidia/spark/examples/agaricus)

## 2. Spark-ETL+XGBoost getting started guides

You can get a small size datasets for each example in the [datasets](../../datasets) folder. 
These datasets are only provided for convenience. In order to test for performance, 
please prepare a larger dataset by following [Preparing Datasets via Notebook](../../datasets/preparing_datasets.md). 
We also provide a larger dataset: [Morgage Dataset (1 GB uncompressed)](https://rapidsai-data.s3.us-east-2.amazonaws.com/spark/mortgage.zip), 
which is used in the guides below.

- Prepare packages and dataset
    - [Scala](../../docs/get-started/xgboost-examples/prepare-package-data/preparation-scala.md)
    - [Python](../../docs/get-started/xgboost-examples/prepare-package-data/preparation-python.md)
- Getting started on on-premises clusters
    - [Standalone cluster for Scala](../../docs/get-started/xgboost-examples/on-prem-cluster/standalone-scala.md)
    - [Standalone cluster for Python](../../docs/get-started/xgboost-examples/on-prem-cluster/standalone-python.md)
    - [YARN for Scala](../../docs/get-started/xgboost-examples/on-prem-cluster/yarn-scala.md)
    - [YARN for Python](../../docs/get-started/xgboost-examples/on-prem-cluster/yarn-python.md)
    - [Kubernetes](../../docs/get-started/xgboost-examples/on-prem-cluster/kubernetes.md)
- Getting started on cloud service providers
    - Amazon AWS
        - [EC2](../../docs/get-started/xgboost-examples/csp/aws/ec2.md)
    - [Databricks](../../docs/get-started/xgboost-examples/csp/databricks/databricks.md)
- Getting started for Jupyter Notebook applications
    - [Apache Toree Notebook for Scala](../../docs/get-started/xgboost-examples/notebook/toree.md)
    - [Jupyter Notebook for Python](../../docs/get-started/xgboost-examples/notebook/python-notebook.md)

These examples use default parameters for demo purposes. For a full list please see "Supported Parameters" 
for [Scala](/app-parameters/supported_xgboost_parameters_scala.md) 
or [Python](/app-parameters/supported_xgboost_parameters_python.md)