<img src="http://developer.download.nvidia.com/notebooks/dlsw-notebooks/tensorrt_torchtrt_efficientnet/nvidia_logo.png" width="110px">

# Distributed Hyperparameter Tuning

## Optuna + Spark + GPUs

These examples demonstrate distributed hyperparameter tuning with [Optuna](https://optuna.readthedocs.io/en/stable/index.html) on Apache Spark, accelerated with [RAPIDS](https://rapids.ai/) on GPU. We showcase how to set up and tune XGBoost on GPU, with deployment on Spark Standalone or Databricks clusters. 

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

Optuna is a lightweight Python library for hyperparameter tuning, integrating state-of-the-art hyperparameter optimization algorithms.  

At a high level, we optimize hyperparameters in three steps (adapted from the [Optuna docs](https://optuna.org/#code_examples)):
1. Wrap model training with an `objective` function that returns a loss metric.
2. In each `trial`, suggest hyperparameters based on previous results.
3. Create a `study` object, which executes the optimization and stores the trial results.

In the distributed setting, each worker gets a copy of the dataset and runs a subset of the trials in parallel (a 'task'), communicating trial results via a shared database. 

We provide **2 notebooks**, with differences in the backend/implementation (see [implementation notes](#implementation-notes) for more details):  

**Using JoblibSpark backend:**
- `joblibspark-xgboost.py`: Uses the [Joblib Spark backend](https://github.com/joblib/joblib-spark) to distribute tasks on the Spark cluster, building off the [Databricks example](https://docs.databricks.com/en/machine-learning/automl-hyperparam-tuning/optuna.html). Each worker reads the full dataset from a specified filepath (e.g., distributed file system). I/O is performed on the worker.

**Using Spark Dataframe:**

- `sparkrapids-xgboost-read-per-worker.py`: Uses Spark dataframes to distribute tasks on the cluster. Each worker reads the full dataset from a specified filepath (e.g., distributed file system). I/O is performed on the worker.
- `sparkrapids-xgboost-copy-and-map.py`: Uses Spark dataframes to distribute tasks on the cluster. Spark reads the dataset from a specified filepath, then duplicates and repartitions it so that each worker task is mapped onto a copy of the dataset. I/O is handled by Spark.

These examples are accelerated on GPU with the [RAPIDS Accelerator](https://nvidia.github.io/spark-rapids/). For more info on the implementations, 

## Running Optuna on Spark Standalone

### Setup database for Optuna

Optuna offers an RDBStorage option which allows for the persistence of experiments
across different machines and processes, thereby enabling Optuna tasks to be distributed.

This guide will walk you through setting up MySQL as the backend for RDBStorage in Optuna.

We highly recommend installing MySQL on the driver node. This setup eliminates concerns
regarding MySQL connectivity between worker nodes and the driver, simplifying the
management of database connections. (For Databricks, the installation is handled by the init script).

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

See the [RAPIDS docs](https://docs.rapids.ai/install/#get-rapids) for version selection.
``` shell
sudo apt install libmysqlclient-dev

conda create -n rapids-24.10 -c rapidsai -c conda-forge -c nvidia  \
    cudf=24.10 cuml=24.10 python=3.10 'cuda-version>=12.0,<=12.5'
conda activate optuna-spark
pip install mysqlclient
pip install optuna joblib joblibspark ipywidgets
```

### Create Optuna database and study

On the driver node, run the following commands to establish a database in MySql.

``` shell
mysql -u optuna_user -p -e "CREATE DATABASE IF NOT EXISTS optuna"
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

After packing the optuna runtime environment, configure your standalone cluster settings. 
This example just creates local cluster with a single GPU worker:
```shell
export SPARK_HOME=/path/to/spark
export SPARK_WORKER_OPTS="-Dspark.worker.resource.gpu.amount=1  \
    -Dspark.worker.resource.gpu.discoveryScript=$SPARK_HOME/examples/src/main/scripts/getGpusResources.sh"
export MASTER=spark://$(hostname):7077; export SPARK_WORKER_INSTANCES=1; export CORES_PER_WORKER=8

${SPARK_HOME}/sbin/start-master.sh; ${SPARK_HOME}/sbin/start-worker.sh -c ${CORES_PER_WORKER} -m 16G ${MASTER}
```

### Configure and run the script

If desired, modify the run scripts to customize the Spark config and the script arguments:  
- `--filepath` (required for XGBoost examples): The absolute path to the WineQuality dataset used in XGBoost examples.
- `--tasks` (optional): Number of Spark tasks to launch, <= parallelism of Spark (default: `2`).
- `--trials` (optional): Number of trials for Optuna to perform (default: `100`).
- `--jobs` (optional, for JoblibSpark only): number of Spark application threads to run in parallel (default: `8`).
- `--localhost` (optional, no argument): Sets the MySQL backend IP to localhost. Included by default in the standalone run scripts.  

Download the dataset:
```shell
wget https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv  \
    -O /path/to/winequality-red.csv
export FILEPATH=/path/to/winequality-red.csv
```

Create and run study:
``` shell
optuna create-study --study-name "optuna-spark-xgboost"  \
    --storage "mysql://optuna_user:optuna_password@localhost/optuna"
export PYSPARK_DRIVER_PYTHON=/path/to/anaconda3/envs/optuna-spark/bin/python
cd standalone
chmod +x run-joblibspark-xgboost.sh
./run-joblibspark-xgboost.sh
```

For `sparkrapids-xgboost` examples, download the [Spark-RAPIDS plugin](https://nvidia.github.io/spark-rapids/docs/download.html) and select the worker-io or spark-io implementation using the SCRIPT environment variable before running:
```shell
SPARK_RAPIDS_VERSION=24.10.1
curl -L https://repo1.maven.org/maven2/com/nvidia/rapids-4-spark_2.12/${SPARK_RAPIDS_VERSION}/rapids-4-spark_2.12-${SPARK_RAPIDS_VERSION}-cuda11.jar -o \
    rapids-4-spark_2.12-${SPARK_RAPIDS_VERSION}.jar
export RAPIDS_JAR=rapids-4-spark_2.12-${SPARK_RAPIDS_VERSION}.jar

cd standalone
# for example, use the read-per-worker implementation:
export SCRIPT=sparkrapids-xgboost-read-per-worker.py
chmod +x run-sparkrapids-xgboost.sh
./run-sparkrapids-xgboost.sh
```

## Running Optuna + Spark on Databricks

### 1. Upload init and Python scripts

- Make sure your [Databricks CLI]((https://docs.databricks.com/en/dev-tools/cli/tutorial.html)) is configured for your Databricks workspace.
- Copy the desired Python script into your Databricks workspace, for example:
    ```shell
    databricks workspace import /path/in/workspace/to/sparkrapids-xgboost-read-per-worker.py  \
        --format AUTO --file sparkrapids-xgboost-read-per-worker.py
    ```
- Copy the init script ```databricks/init_optuna_xgboost.sh```:
    ```shell
    databricks workspace import /path/in/workspace/to/init_optuna_xgboost.sh  \
        --format AUTO --file databricks/init_optuna_xgboost.sh
    ```
- (For XGBOOST example): Upload the [Wine Qualities](https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv) dataset via the Databricks CLI:
    ```shell
    wget https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv  \
        -O winequality-red.csv
    databricks fs mkdir dbfs:/FileStore/datasets/
    databricks fs cp winequality-red.csv dbfs:/FileStore/datasets/winequality-red.csv
     ```

### 2. Create cluster

Run the cluster startup script, which is configured to create an 8 node GPU cluster:
```shell
export INIT_PATH=/path/in/workspace/to/init_optuna_xgboost.sh
cd databricks
chmod +x start_cluster.sh
./start_cluster.sh
```

Or, create a cluster via the web UI:
- Go to `Compute > Create compute` and set the desired cluster settings.    
- Under `Advanced Options > Init Scripts`, upload the init script from your workspace.
- Under `Advanced Options > Spark > Environment variables`, set `LIBCUDF_CUFILE_POLICY=OFF`. 
- Make sure to use a GPU cluster and include task GPU resources.

The init script will install the required libraries on all nodes, including rapids/cuml for data operations on GPU. On the driver, it will setup the MySQL server backend and create an Optuna study referencing the server. 

### 3. Run Script

To run the script, you can either:
- **Run interactively**: [convert the script into a Databricks notebook](https://docs.databricks.com/en/notebooks/notebook-export-import.html#import-a-file-and-convert-it-to-a-notebook), attach to the cluster, and run on the web UI. 
- **Run as a Databricks job**: submit a job via the CLI; replace the \<fields\> as needed. The cluster ID can be retrieved with ```databricks clusters list```:
  ```shell
  databricks jobs submit --json '{
    "name": "Optuna-Spark-XGBoost",
    "tasks": [
      {
        "existing_cluster_id": "<cluster_id>",
        "task_key": "spark_python_task",
        "spark_python_task": {
          "python_file": "</path/in/workspace/to/sparkrapids-xgboost-read-per-worker.py>",
          "parameters": [
            "--filepath", "/dbfs/FileStore/datasets/winequality-red.csv",
            "--trials", "100",
            "--tasks", "8"
          ]
        }
      }
    ]
  }'
  ```
  


## How does it work?

The Optuna tasks will be serialized into bytes and distributed to Spark workers to run. The Optuna task on the executor side that loads the Optuna study from RDBStorage, and then runs its set of trials.

During tuning, the Optuna tasks send intermediate results back to RDBStorage to persist, and ask for the parameters from RDBStorage sampled by Optuna on the driver to run next.

Using JoblibSpark, each Optuna task is a Spark application that has only 1 job, 1 stage, 1 task, and the Spark application will be submitted on the local threads. Here the parameter `n_jobs` configures the Spark backend to limit how many Spark applications are submitted at the same time.  

Thus Optuna with JoblibSpark uses Spark application level parallelism, rather than task-level parallelism. So for the XGBoost case, we first need to ensure that the single XGBoost task can run on a single node without any CPU/GPU OOM.  

Application parallelism with JoblibSpark:  

![Optuna on JoblibSpark](images/optuna.svg)

### Implementation Notes

###### Data passing in sparkrapids-xgboost:
Since each worker requires the full dataset to perform hyperparameter tuning, there are two strategies to enable this:
  - In joblibspark-xgboost.py, sparkrapids-xgboost-read-per-worker.py: **each worker reads the dataset** from the filepath once the task has begun. In practice, this requires the dataset to be written to a distributed file system accessible to all workers prior to tuning. 
  - In sparkrapids-xgboost-copy-and-map.py: Spark reads the dataset and **creates a copy of the dataset for each worker**, then maps the tuning task onto each copy. In practice, this enables the code to be chained to other Dataframe operations (e.g. ETL stages) without the intermediate step of writing to DBFS, at the cost of memory overhead during duplication.
    - To do this, we coalesce the input Dataframe to a single partition, and recursively self-union until we have the desired number of copies (number of workers). Thus each partition will contain a duplicate of the entire dataset, and the Optuna task can be mapped directly onto the partitions.


###### Misc:
- Please be aware that Optuna studies will continue where they left off from previous trials; delete and recreate the study if you would like to start anew.
- Optuna in distributed mode is **non-deterministic** (see [this link](https://optuna.readthedocs.io/en/stable/faq.html#how-can-i-obtain-reproducible-optimization-results)), as trials are executed asynchronously by executors. Deterministic behavior can be achieved using Spark barriers to coordinate reads/writes to the database.
- Reading data with GPU using cuDF requires disabling [GPUDirect Storage](https://docs.rapids.ai/api/cudf/nightly/user_guide/io/io/#magnum-io-gpudirect-storage-integration), i.e., setting the environment variable `LIBCUDF_CUFILE_POLICY=OFF`, to be compatible with the Databricks file system. Without GDS, cuDF will use a CPU bounce buffer when reading files, but all parsing and decoding will still be accelerated by the GPU. 
