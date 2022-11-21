# spark-rapids-examples

This is the [RAPIDS Accelerator for Apache Spark](https://nvidia.github.io/spark-rapids/) examples repo.
RAPIDS Accelerator for Apache Spark accelerates Spark applications with no code changes.
You can download the latest version of RAPIDS Accelerator [here](https://nvidia.github.io/spark-rapids/docs/download.html).
This repo contains examples and applications that showcases the performance and benefits of using 
RAPIDS Accelerator in data processing and machine learning pipelines. 
There are broadly four categories of examples in this repo: 
1. [SQL/Dataframe](./examples/SQL+DF-Examples) 
2. [Spark XGBoost](./examples/XGBoost-Examples) 
3. [Deep Learning/Machine Learning](./examples/ML+DL-Examples) 
4. [RAPIDS UDF](./examples/UDF-Examples)
5. [Databricks Tools demo notebooks](./tools/databricks)

For more information on each of the examples please look into respective categories.

Here is the list of notebooks in this repo:

|   | Category  | Notebook Name | Description
| ------------- | ------------- | ------------- | -------------
| 1 | SQL/DF | Microbenchmark | Spark SQL operations such as expand, hash aggregate, windowing, and cross joins with up to 20x performance benefits
| 2 | SQL/DF | Customer Churn | Data federation for modeling customer Churn with a sample telco customer data
| 3 | XGBoost | Agaricus (Scala) | Uses XGBoost classifier function to create model that can accurately differentiate between edible and poisonous mushrooms with the [agaricus dataset](https://archive.ics.uci.edu/ml/datasets/mushroom)
| 4 | XGBoost | Mortgage (Scala) | End-to-end ETL + XGBoost example to predict mortgage default with [Fannie Mae Single-Family Loan Performance Data](https://capitalmarkets.fanniemae.com/credit-risk-transfer/single-family-credit-risk-transfer/fannie-mae-single-family-loan-performance-data)
| 5 | XGBoost | Taxi (Scala) | End-to-end ETL + XGBoost example to predict taxi trip fare amount with [NYC taxi trips data set](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
| 6 | ML/DL | Criteo Training | ETL and deep learning training of the Criteo 1TB Click Logs dataset
| 7 | ML/DL | PCA End-to-End | Spark MLlib based PCA example to train and transform with a synthetic dataset
| 8 | UDF | cuSpatial - Point in Polygon | Spark cuSpatial example for Point in Polygon function using NYC Taxi pickup location dataset

Here is the list of Apache Spark applications (Scala and PySpark) that 
can be built for running on GPU with RAPIDS Accelerator in this repo:

|   | Category  | Notebook Name | Description
| ------------- | ------------- | ------------- | -------------
| 1 | XGBoost | Agaricus (Scala) | Uses XGBoost classifier function to create model that can accurately differentiate between edible and poisonous mushrooms with the [agaricus dataset](https://archive.ics.uci.edu/ml/datasets/mushroom)
| 2 | XGBoost | Mortgage (Scala) | End-to-end ETL + XGBoost example to predict mortgage default with [Fannie Mae Single-Family Loan Performance Data](https://capitalmarkets.fanniemae.com/credit-risk-transfer/single-family-credit-risk-transfer/fannie-mae-single-family-loan-performance-data)
| 3 | XGBoost | Taxi (Scala) | End-to-end ETL + XGBoost example to predict taxi trip fare amount with [NYC taxi trips data set](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
| 4 | ML/DL | PCA End-to-End | Spark MLlib based PCA example to train and transform with a synthetic dataset
| 5 | UDF | cuSpatial - Point in Polygon | Spark cuSpatial example for Point in Polygon function using NYC Taxi pickup location dataset
| 6 | UDF | URL Decode | Decodes URL-encoded strings using the [Java APIs of RAPIDS cudf](https://docs.rapids.ai/api/cudf-java/stable/)
| 7 | UDF | URL Encode | URL-encodes strings using the [Java APIs of RAPIDS cudf](https://docs.rapids.ai/api/cudf-java/stable/)
| 8 | UDF | [CosineSimilarity](./examples/UDF-Examples/RAPIDS-accelerated-UDFs/src/main/java/com/nvidia/spark/rapids/udf/java/CosineSimilarity.java) | Computes the cosine similarity between two float vectors using [native code](./examples/UDF-Examples/RAPIDS-accelerated-UDFs/src/main/cpp/src)
| 9 | UDF | [StringWordCount](./examples/UDF-Examples/RAPIDS-accelerated-UDFs/src/main/java/com/nvidia/spark/rapids/udf/hive/StringWordCount.java)  | Implements a Hive simple UDF using [native code](./examples/UDF-Examples/RAPIDS-accelerated-UDFs/src/main/cpp/src) to count words in strings
