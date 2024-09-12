# Running Spark Optuna on Databricks

**1. Upload init and Python scripts**

- Login to your Databricks workspace.
- Go to the `Workspace` section on the left sidebar.
- Click on `Create > Notebook`. Copy the desired Python script ```optuna-mysql-spark-databricks.py``` or ```optuna-mysql-xgboost-spark-databricks.py``` into the notebook.
    - **Note**: we also provide the script ```optuna-mysql-spark-databricks-rdd.py```, which uses native Spark RDDs rather than Joblib-Spark to distribute trials across the cluster. This method avoids the need for Joblib installation and uses pure Spark, and may be preferred if integrating this code into Spark pipelines. 
- Click on `Create > File` and copy in the desired init script ```init_optuna.sh``` or ```init_optuna_xgboost.sh```. 
    - Alternatively, you can copy in files using the Databricks CLI, for example:
     ```shell
     databricks workspace import /path/to/directory/in/workspace --format AUTO --file init_optuna.sh
     ```
- (For XGBOOST example): Upload the [Wine Qualities](https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv) dataset via the Databricks CLI:
    ```shell
    wget https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv -O winequality-red.csv
    databricks fs mkdir dbfs:/FileStore/datasets/
    databricks fs cp winequality-red.csv dbfs:/FileStore/datasets/winequality-red.csv
     ```

**2. Create cluster**

- Go to `Compute > Create compute` and set the desired cluster settings.    
- Under `Advanced Options > Init Scripts`, upload the init script from your workspace.
- Under `Advanced Options > Spark > Environment variables`, set `LIBCUDF_CUFILE_POLICY=OFF`. 
- (For XGBOOST example): Make sure to use a GPU cluster, and set the Spark config value for `spark.task.resource.gpu.amount`.

The init script will install the required libraries on all nodes, including rapids/cuml for array operations on GPU. On the driver, it will setup the MySQL server backend and create an Optuna study referencing the server. See [here](/optuna?ref_type=heads#how-does-it-work) for details on how the cluster parallelism works.

**3. Run Notebook**

- Attach the Python Notebook to the cluster and run.

# Implementation Notes

- Optuna in distributed mode is **non-deterministic** (see [this link](https://optuna.readthedocs.io/en/stable/faq.html#how-can-i-obtain-reproducible-optimization-results)), as trials are executed asynchronously by executors. Deterministic behavior would require synchronizing executor reads/writes to the database, and would require changes to Optuna internals. 
- Reading data with GPU using cuDF requires disabling [GPUDirect Storage](https://docs.rapids.ai/api/cudf/nightly/user_guide/io/io/#magnum-io-gpudirect-storage-integration), i.e., setting the environment variable `LIBCUDF_CUFILE_POLICY=OFF`, to be compatible with the Databricks file system. Without GDS, cuDF will use a CPU bounce buffer when reading files, but all parsing and decoding will still be accelerated by the GPU. 
- 
