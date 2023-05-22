# Spark DL Training and Inference

This example is based on the [distributed training example](https://docs.databricks.com/machine-learning/train-model/distributed-training/spark-pytorch-distributor.html#end-to-end-distributed-training-on-databricks-notebook) from Databricks, with additional code to demonstrate:
- how to access datasets on a distributed store, like S3, using Petastorm and/or NVTabular.
- how to conduct parallel inference of a DL model on a Spark cluster, using a new API introduced in Spark 3.4.

## Run example in Databricks
- Create an AWS S3 bucket to hold the dataset
- Create an AWS IAM User with access keys for the S3 bucket
- Add the AWS secrets to the [Databricks Secrets](https://docs.databricks.com/security/secrets/secrets.html) store using the [databricks CLI](https://docs.databricks.com/dev-tools/cli/index.html)
  ```
  databricks secrets create-scope --scope aws-s3
  databricks secrets put --scope aws-s3 --key aws-access-key
  databricks secrets put --scope aws-s3 --key aws-secret-key
  ```
  **NOTE**: Do not hard-code the values of the secrets anywhere and avoid accidental logging of the values.

- Copy the [init.sh](./init.sh) script to DBFS using the
  ```
  databricks fs cp init.sh dbfs:/path/to/init.sh
  ```
- Create a multi-node Databricks cluster with the following configuration:
  - Databricks Runtime Versions: `13.1 ML GPU Runtime`
  - Worker type: `g4dn.xlarge` (x2)
  - Driver type: `same as worker`
  - Advanced options -> Init Scripts: `dbfs:/path/to/init.sh`
- Start the cluster
- Import the notebook into your Databricks workspace
- Attach the cluster to your notebook and run the notebook
