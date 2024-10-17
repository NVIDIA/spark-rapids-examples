# Optuna on Spark + GPUs

This repo demonstrates distributed hyperparameter tuning with [Optuna](https://optuna.readthedocs.io/en/stable/index.html) on Apache Spark using the MySQL storage backend, accelerated with [RAPIDS](https://rapids.ai/) data processing on GPU. It contains examples that showcase how to set up and tune XGBoost on GPU, with deployment on Spark Standalone or Databricks clusters. 

## Contents:
- [Overview](#overview)
- [Running Optuna on Spark Standalone](#running-optuna-on-spark-standalone)
  - [Setup Database for Optuna](#setup-database-for-optuna)
  - [Setup Optuna Python Environment](#setup-optuna-python-environment)
  - [Create Optuna Database and Study](#create-optuna-database-and-study)
  - [Pack the Optuna Runtime Environment and Run](#pack-the-optuna-runtime-environment-and-start-cluster)
  - [Configure and run the script](#configure-and-run-the-script)
- [Running Optuna + Spark on Databricks](#running-optuna--spark-on-databricks)
  - [Upload Init and Python Scripts](#1-upload-init-and-python-scripts)
  - [Create Cluster](#2-create-cluster)
  - [Run Script](#3-run-script)
- [How Does it Work?](#how-does-it-work)
  - [Implementation Notes](#implementation-notes)

---

## Overview

**Please refer to the instructions below to properly setup your environment and run the script for [Spark Standalone](#running-optuna-on-spark-standalone) or [Databricks](#running-optuna--spark-on-databricks).**  
 The Python scripts take the following arguments:

- `--filepath` (required for XGBoost examples): The absolute path (locally or on /dbfs/) to the dataset used in XGBoost examples.
- `--tasks` (optional): Number of Spark tasks to launch, <= parallelism of Spark (default: `2`).
- `--trials` (optional): Number of trials for Optuna to perform (default: `100`).
- `--jobs` (optional): Number of Spark application threads to run in parallel (default: `8`).

We provide the following scripts under `/examples/`:  
- `optuna-mysql-spark.py`: Demonstrates a simple study on CPU to minimize a quadratic function using the MySQL backend and Joblib Spark to distribute tasks.
- `optuna-mysql-xgboost-spark.py`: Demonstrates a regression study to tune XGBoost on GPU to predict red wine quality, adapted from [this example](https://forecastegy.com/posts/xgboost-hyperparameter-tuning-with-optuna/), using the MySQL backend and Joblib Spark to distribute tasks.
- `optuna-mysql-xgboost-spark-rdd.py`: Demonstrates a regression study to tune XGBoost on GPU to predict red wine quality as above, but uses native Spark RDDs rather than Joblib-Spark to distribute trials across the cluster. 

## Running Optuna on Spark Standalone

### Setup database for Optuna

Optuna offers an RDBStorage option which allows for the persistence of experiments
across different machines and processes, thereby enabling Optuna tasks to be distributed.

This guide will walk you through setting up MySQL as the backend for RDBStorage in Optuna.

We highly recommend installing MySQL on the driver node. This setup eliminates concerns
regarding MySQL connectivity between worker nodes and the driver, simplifying the
management of database connections.

1. Install MySql

``` shell
sudo apt install mysql-server
```

2. Configure MySql

in `/etc/mysql/mysql.conf.d/mysqld.cnf`

``` shell
bind-address    = THE_DRIVER_HOST_IP
mysqlx-bind-address = THE_DRIVER_HOST_IP
```

3. Restart MySql

``` shell
sudo systemctl restart mysql.service
```

4. Setup user

```shell
sudo mysql
```

``` mysql
mysql> CREATE USER 'optuna_user'@'%' IDENTIFIED BY 'optuna_password';
Query OK, 0 rows affected (0.01 sec)

mysql> GRANT ALL PRIVILEGES ON *.* TO 'optuna_user'@'%' WITH GRANT OPTION;
Query OK, 0 rows affected (0.01 sec)

mysql> FLUSH PRIVILEGES;
Query OK, 0 rows affected (0.01 sec)

mysql> EXIT;
Bye
```

Troubleshooting:  
> If you encounter   
`"ERROR 2002 (HY000): Can't connect to local MySQL server through socket '/tmp/mysql.sock' (2)"`,  
try the command:  
`ln -s /var/run/mysqld/mysqld.sock /tmp/mysql.sock`

### Setup Optuna Python environment

``` shell
sudo apt install libmysqlclient-dev

conda create -n optuna-spark -c rapidsai -c conda-forge -c nvidia  \
    rapids=24.08 python=3.10 'cuda-version>=12.0,<=12.5' # see https://docs.rapids.ai/install/#get-rapids for version selection
conda activate optuna-spark
pip install mysqlclient
pip install optuna joblib joblibspark
```

### Create Optuna database and study

On the driver node, run the following commands to establish a database in MySql and create
an Optuna study.

``` shell
mysql -u optuna_user -p -e "CREATE DATABASE IF NOT EXISTS optuna"
optuna create-study --study-name "optuna-spark" --storage "mysql://optuna_user:optuna_password@localhost/optuna"
```

Troubleshooting:  
> If you encounter   
`ImportError: ... GLIBCXX_3.4.32' not found`,  
you can run the following to override Conda to use your system's `libstdc++.so.6`:  
`export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6`

### Pack the Optuna runtime environment and start cluster

``` shell
pip install conda-pack
conda pack -f -o optuna-env.tar.gz
```

After packing the optuna runtime environment, configure your standalone cluster settings. Make sure to download the [GPU discovery script](https://github.com/apache/spark/blob/master/examples/src/main/scripts/getGpusResources.sh) and specify the absolute path to it.
This example just creates local cluster with a single GPU worker:
```shell
export SPARK_HOME=/path/to/spark
export SPARK_WORKER_OPTS="-Dspark.worker.resource.gpu.amount=1 -Dspark.worker.resource.gpu.discoveryScript=/path/to/getGpusResources.sh"
export MASTER=spark://$(hostname):7077; export SPARK_WORKER_INSTANCES=1; export CORES_PER_WORKER=8

${SPARK_HOME}/sbin/start-master.sh; ${SPARK_HOME}/sbin/start-worker.sh -c ${CORES_PER_WORKER} -m 16G ${MASTER}
```

### Configure and run the script

Modify the run scripts to customize the Spark config and task/job parallelism.  
You can then run the simple demo of Optuna on Spark: 

```shell
export PYSPARK_DRIVER_PYTHON=/path/to/anaconda3/envs/optuna-spark/bin/python
cd standalone
chmod +x run-optuna-spark.sh
./run-optuna-spark.sh
```

...or the Optuna XGBoost GPU example:

``` shell
optuna create-study --study-name "optuna-spark-xgboost" --storage "mysql://optuna_user:optuna_password@localhost/optuna"
export PYSPARK_DRIVER_PYTHON=/path/to/anaconda3/envs/optuna-spark/bin/python
cd standalone
chmod +x run-optuna-spark-xgboost.sh
./run-optuna-spark-xgboost.sh
```

Please be aware that studies will continue where they left off; delete and recreate the study if you would like to start anew.


## Running Optuna + Spark on Databricks

### 1. Upload init and Python scripts

- Make sure your Databricks CLI is configured for your Databricks workspace. See the [tutorial](https://docs.databricks.com/en/dev-tools/cli/tutorial.html) for setup. 
- Copy the desired Python script into your Databricks workspace, for example:
    ```shell
    databricks workspace import /path/to/directory/in/workspace --format AUTO --file optuna-mysql-xgboost-spark.py
    ```
- Copy the corresponding init script ```databricks/init_optuna.sh``` or ```databricks/init_optuna_xgboost.sh```, for example:
    ```shell
    databricks workspace import /path/to/directory/in/workspace --format AUTO --file databricks/init_optuna_xgboost.sh
    ```
- (For XGBOOST example): Upload the [Wine Qualities](https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv) dataset via the Databricks CLI:
    ```shell
    wget https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv -O winequality-red.csv
    databricks fs mkdir dbfs:/FileStore/datasets/
    databricks fs cp winequality-red.csv dbfs:/FileStore/datasets/winequality-red.csv
     ```

### 2. Create cluster

- Go to `Compute > Create compute` and set the desired cluster settings.    
- Under `Advanced Options > Init Scripts`, upload the init script from your workspace.
- Under `Advanced Options > Spark > Environment variables`, set `LIBCUDF_CUFILE_POLICY=OFF`. 
- (For XGBOOST example): Make sure to use a GPU cluster, and set the Spark config value for `spark.task.resource.gpu.amount`.

The init script will install the required libraries on all nodes, including rapids/cuml for data operations on GPU. On the driver, it will setup the MySQL server backend and create an Optuna study referencing the server. 

### 3. Run Script

To run the script, you can either:
- **Run interactively**: copy the script into a Databricks notebook, attach to the cluster, and run on the web UI. 
- **Run as a Databricks job**: create a job via the CLI, for example:
  ```shell
  databricks jobs create --json '{
    "name": "Optuna-MySQL-XGBoost-Spark",
    "tasks": [
      {
        "existing_cluster_id": "<cluster_id>",
        "task_key": "spark_python_task",
        "spark_python_task": {
          "python_file": "/path/in/workspace/to/script.py",
          "parameters": [
            "--filepath", "/dbfs/Filestore/datasets/winequality-red.csv",
            "--trials", "100",
            "--tasks", "2",
            "--jobs", "8"
          ]
        }
      }
    ]
  }'
  ```
  - The cluster ID can be retrieved with ```databricks clusters list```. 
  - The job can be viewed and run on the Databricks `Workflows` panel or using `databricks jobs run-now` in the CLI.



## How does it work?

![](images/optuna.svg)

The Optuna tasks will be serialized into bytes and distributed to Spark workers
to run. So it's the Optuna task on the executor side that loads the optuna study from RDBStorage, and then
runs the tuning.

During tuning, the Optuna tasks need to send intermediate results back to RDBStorage to persist,
and ask for the parameters from RDBStorage sampled by Optuna on the driver to run next.

Using JoblibSpark, each Optuna task is a Spark application that has only 1 job, 1 stage, 1 task, and the Spark application
will be submitted on the local threads. So here we can use `n_jobs` when configuring the Spark backend
to limit at most how many Spark applications can be submitted at the same time.  

Thus Optuna with JoblibSpark uses Spark application level parallelism, rather than task-level parallelism. So for the XGBoost case, we first need to ensure that the single XGBoost task can run on a single node without any CPU/GPU OOM.

### Implementation Notes

- Please be aware that Optuna studies will continue where they left off; delete and recreate the study if you would like to start anew.
- Optuna in distributed mode is **non-deterministic** (see [this link](https://optuna.readthedocs.io/en/stable/faq.html#how-can-i-obtain-reproducible-optimization-results)), as trials are executed asynchronously by executors. Deterministic behavior would require synchronizing executor reads/writes to the database, which are handled internally by Optuna.
- Reading data with GPU using cuDF requires disabling [GPUDirect Storage](https://docs.rapids.ai/api/cudf/nightly/user_guide/io/io/#magnum-io-gpudirect-storage-integration), i.e., setting the environment variable `LIBCUDF_CUFILE_POLICY=OFF`, to be compatible with the Databricks file system. Without GDS, cuDF will use a CPU bounce buffer when reading files, but all parsing and decoding will still be accelerated by the GPU. 
