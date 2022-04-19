Get Started with XGBoost4J-Spark with Spylon Kernel Jupyter Notebook
===================================================================

This is a getting started guide to XGBoost4J-Spark using a [Spylon Kernel](https://pypi.org/project/spylon-kernel/) Jupyter notebook. 
At the end of this guide, the reader will be able to run a sample notebook that runs on NVIDIA GPUs.

Before you begin, please ensure that you have setup 
a [Spark Standalone Cluster](/docs/get-started/xgboost-examples/on-prem-cluster/standalone-scala.md).

It is assumed that the `SPARK_MASTER` and `SPARK_HOME` environment variables are defined and point to the Spark Master URL, 
and the home directory for Apache Spark respectively.

1. Install Jupyter Notebook with spylon-kernel.
   ``` bash
   # Install notebook and spylon-kernel (Scala kernel for Jupyter Notebook), https://pypi.org/project/spylon-kernel/
   # You can use spylon-kernel as Scala kernel for Jupyter Notebook. Do this when you want to work with Spark in Scala with a bit of Python code mixed in.
   RUN pip3 install jupyter notebook spylon-kernel
   RUN python -m spylon_kernel install
   # Latest version breaks nbconvert: https://github.com/ipython/ipykernel/issues/422
   RUN pip3 install ipykernel==5.1.1
   ```
2. Start Jupyter Notebook. 

You can debug from webUI http://your_ip:your_port with your password.
    
    ``` bash
    export JUPYTER_CONFIG_FILE=~/.jupyter/jupyter_notebook_config.py
    
    rm -rf `dirname $JUPYTER_CONFIG_FILE` && mkdir -p `dirname $JUPYTER_CONFIG_FILE` && echo """
    c.NotebookApp.ip='*'
    c.NotebookApp.password = your_hashed_password
    c.NotebookApp.password = your_password 
    c.NotebookApp.open_browser = False
    c.NotebookApp.port = your_port
    """ > $JUPYTER_CONFIG_FILE
 
    jupyter notebook --allow-root --notebook-dir=$WORKSPACE --config=$JUPYTER_CONFIG_FILE &
    ```
3. Prepare packages and dataset.

    Make sure you have prepared the necessary packages and dataset by following this [guide](/docs/get-started/xgboost-examples/prepare-package-data/preparation-scala.md)

4. Run scala notebook (e.g. [mortgage-gpu.ipynb](/examples/Spark-ETL+XGBoost(splitting)/mortgage/notebooks/scala/mortgage-gpu.ipynb))

    ``` bash
    # Suppose your Scala file is $WORKSPACE/mortgage-gpu.ipynb
 
    jupyter nbconvert --to notebook --stdout --execute $WORKSPACE/mortgage-gpu.ipynb
     
    # -------you will see output looks like ----------------
    # { 
    #   "cells": [
    #   {
    #    "cell_type": "code",
    #    "execution_count": 1,
    #    "id": "5ca1ae16",
    #    "metadata": {
    #     ........
    #     ........
    #     ........
    #   "language_info": {
    #    "codemirror_mode": "text/x-scala",
    #    "file_extension": ".scala",
    #    "help_links": [
    #     {
    #      "text": "MetaKernel Magics",
    #      "url": "https://metakernel.readthedocs.io/en/latest/source/README.html"
    #     }
    #    ],
    #    "mimetype": "text/x-scala",
    #    "name": "scala",
    #    "pygments_lexer": "scala",
    #    "version": "0.4.1"
    #   }
    #  },
    #  "nbformat": 4,
    #  "nbformat_minor": 5
    # }
    ```
    You can also run python notebook with Spylon Kernel
    ``` bash
    # restart Jupyter Notebook
  
    export PYSPARK_DRIVER_PYTHON=jupyter
    export PYSPARK_DRIVER_PYTHON_OPTS="notebook --allow-root --notebook-dir=$WORKSPACE --config=$JUPYTER_CONFIG_FILE"
    pyspark &
     
    # Suppose your python file is $WORKSPACE/mortgage-gpu.ipynb
    jupyter nbconvert --to notebook--stdout --execute $WORKSPACE/mortgage-gpu.ipynb
    ```
   
5. Launch ETL Part 
- Mortgage ETL Notebook: [Scala](/examples/Spark-ETL+XGBoost(splitting)/mortgage/notebooks/scala/mortgage-ETL.ipynb) or
  [Python](/examples/Spark-ETL+XGBoost(splitting)/mortgage/notebooks/python/MortgageETL.ipynb)
- Taxi ETL Notebook: [Scala](/examples/Spark-ETL+XGBoost(splitting)/taxi/notebooks/scala/taxi-ETL.ipynb) or
  [Python](/examples/Spark-ETL+XGBoost(splitting)/taxi/notebooks/python/taxi-ETL.ipynb)
- Note: Agaricus does not have ETL part.
   
6. Launch XGBoost Part
- Mortgage XGBoost Notebook: [Scala](/examples/Spark-ETL+XGBoost(splitting)/mortgage/notebooks/scala/mortgage-gpu.ipynb) or
  [Python](/examples/Spark-ETL+XGBoost(splitting)/mortgage/notebooks/python/mortgage-gpu.ipynb)
- Taxi XGBoost Notebook: [Scala](/examples/Spark-ETL+XGBoost(splitting)/taxi/notebooks/scala/taxi-gpu.ipynb) or 
  [Python](/examples/Spark-ETL+XGBoost(splitting)/taxi/notebooks/python/taxi-gpu.ipynb)
- Agaricus XGBoost Notebook: [Scala](/examples/Spark-ETL+XGBoost(splitting)/agaricus/notebooks/scala/agaricus-gpu.ipynb) or
  [Python](/examples/Spark-ETL+XGBoost(splitting)/agaricus/notebooks/python/agaricus-gpu.ipynb)