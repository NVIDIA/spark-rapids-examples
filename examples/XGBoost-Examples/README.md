# Spark-ETL+XGBoost examples

## 1. Prepare the dataset Spark-ETL+XGBoost examples codes

You can get a small size dataset for each example in the [datasets](../../datasets) folder. 
These datasets are only provided for convenience. In order to test for performance, 
please prepare packages and download a larger dataset by following links:
- Prepare packages and datasets
    - [Scala](../../docs/get-started/xgboost-examples/prepare-package-data/preparation-scala.md)
    - [Python](../../docs/get-started/xgboost-examples/prepare-package-data/preparation-python.md)

We provided a [Morgage Dataset (1 GB uncompressed)](https://rapidsai-data.s3.us-east-2.amazonaws.com/spark/mortgage.zip), 
which is used in the Mortgage example below for quick download.

## 2. Spark-ETL+XGBoost getting started guides
We have three examples to demonstrate the Spark ETL + XGBoost process: Mortgage, Taxi and Agaricus(Note: Agaricus example has 
a very simple ETL part which is just a filter, so we combined the ETL part into XGBoost part).
Here are 2 ways to get started:
- Running Spark jobs
- Notebooks

### 1. Getting Started by running Spark jobs


Below are source codes for the example Spark jobs:
- Mortgage: [Scala](../../examples/XGBoost-Examples/mortgage/scala/src/com/nvidia/spark/examples/mortgage), [Python](../../examples/XGBoost-Examples/mortgage/python/com/nvidia/spark/examples/mortgage)
- Taxi: [Scala](../../examples/XGBoost-Examples/taxi/scala/src/com/nvidia/spark/examples/taxi), [Python](../../examples/XGBoost-Examples/taxi/python/com/nvidia/spark/examples/taxi)
- Agaricus: [Scala](../../examples/XGBoost-Examples/agaricus/scala/src/com/nvidia/spark/examples/agaricus), [Python](../../examples/XGBoost-Examples/agaricus/python/com/nvidia/spark/examples/agaricus)


Please follow below steps to run the example Spark jobs in different Spark environments:
- Getting started on on-premises clusters
    - [Standalone cluster for Scala](../../docs/get-started/xgboost-examples/on-prem-cluster/standalone-scala.md)
    - [Standalone cluster for Python](../../docs/get-started/xgboost-examples/on-prem-cluster/standalone-python.md)
    - [YARN for Scala](../../docs/get-started/xgboost-examples/on-prem-cluster/yarn-scala.md)
    - [YARN for Python](../../docs/get-started/xgboost-examples/on-prem-cluster/yarn-python.md)
    - [Kubernetes](../../docs/get-started/xgboost-examples/on-prem-cluster/kubernetes-scala.md)
- Getting started on cloud service providers
    - Amazon AWS
        - [EC2](../../docs/get-started/xgboost-examples/csp/aws/ec2.md)
    - [Databricks](../../docs/get-started/xgboost-examples/csp/databricks/databricks.md)
  
### 2. Getting Started by using Notebooks


Below are the example notebooks:
- Mortgage: [Scala](../../examples/XGBoost-Examples/mortgage/notebooks/scala), [Python](../../examples/XGBoost-Examples/mortgage/notebooks/python)
- Taxi: [Scala](../../examples/XGBoost-Examples/taxi/notebooks/scala), [Python](../../examples/XGBoost-Examples/taxi/notebooks/python)
- Agaricus: [Scala](../../examples/XGBoost-Examples/agaricus/notebooks/scala), [Python](../../examples/XGBoost-Examples/agaricus/notebooks/python)
- Churn: [Python](../SQL+DF-Examples/customer-churn/notebooks/python)

Please follow below steps to run the example notebooks in different notebook environments:
- Getting started for Jupyter Notebook applications
    - [Apache Toree Notebook for Scala](../../docs/get-started/xgboost-examples/notebook/toree.md)
    - [Jupyter Notebook with spylon kernel](../../docs/get-started/xgboost-examples/notebook/spylon.md)
    - [Jupyter Notebook for Python](../../docs/get-started/xgboost-examples/notebook/python-notebook.md)
  
For the CrossValidator job, we need to set `spark.task.resource.gpu.amount=1` to allow only 1 training task running on 1 GPU(executor),
otherwise the customized CrossValidator may schedule more than 1 
xgboost training tasks into one executor simultaneously and trigger 
[issue-131](https://github.com/NVIDIA/spark-rapids-examples/issues/131).

These examples use default parameters for demo purposes. For a full list please see "Supported Parameters"
for [Scala](../../examples/XGBoost-Examples/app-parameters/supported_xgboost_parameters_scala.md) 
or [Python](../../examples/XGBoost-Examples/app-parameters/supported_xgboost_parameters_python.md).