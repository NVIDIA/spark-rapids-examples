# Spark DL Inference Using External Frameworks

Example notebooks for the [predict_batch_udf](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.functions.predict_batch_udf.html#pyspark.ml.functions.predict_batch_udf) function introduced in Spark 3.4.

## Overview

This directory contains notebooks for each DL framework (based on their own published examples).  The goal is to demonstrate how models trained and saved on single-node machines can be easily used for parallel inferencing on Spark clusters.

For example, a basic model trained in TensorFlow and saved on disk as "mnist_model.keras" can be used in Spark as follows:
```
import numpy as np
from pyspark.sql.functions import predict_batch_udf
from pyspark.sql.types import ArrayType, FloatType

def predict_batch_fn():
    import tensorflow as tf
    model = tf.keras.models.load_model("/path/to/mnist_model.keras")
    def predict(inputs: np.ndarray) -> np.ndarray:
        return model.predict(inputs)
    return predict

mnist = predict_batch_udf(predict_batch_fn,
                          return_type=ArrayType(FloatType()),
                          batch_size=1024,
                          input_tensor_shapes=[[784]])

df = spark.read.parquet("mnist_data")
predictions = df.withColumn("preds", mnist("data")).collect()
```

In this simple case, the `predict_batch_fn` will use TensorFlow APIs to load the model and return a simple `predict` function which operates on numpy arrays.  The `predict_batch_udf` will automatically convert the Spark DataFrame columns to the expected numpy inputs. For Huggingface, use the respective environment for the model in the example. 

All notebooks have been saved with sample outputs for quick browsing.

## Running the Notebooks

If you want to run the notebooks yourself, please follow these instructions. 

**Notes**: 
- Please create separate environments for PyTorch/Tensorflow to avoid conflicts between the CUDA libraries bundled with their respective versions. The Huggingface examples will have a _torch or _tf suffix to specify the environment used.
- For demonstration purposes, these examples just use a local Spark Standalone cluster with a single executor, but you should be able to run them on any distributed Spark cluster.
- The notebooks can also be run on your local machine in any Jupyter environment, and will default to using a local Spark Session. 
```
# for pytorch:
conda create -n spark-dl-torch
conda activate spark-dl-torch
# for tensorflow:
conda create -n spark-dl-tf
conda activate spark-dl-tf

# install dependencies
pip install -r requirements.txt

# for pytorch:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install sentence_transformers sentencepiece
# for tensorflow:
pip install tensorflow[and-cuda] tf-keras

# setup environment variables
export SPARK_HOME=/path/to/spark
export MASTER=spark://$(hostname):7077
export SPARK_WORKER_INSTANCES=1
export CORES_PER_WORKER=8
export PYSPARK_DRIVER_PYTHON=jupyter
export PYSPARK_DRIVER_PYTHON_OPTS='lab'

# start spark standalone cluster
${SPARK_HOME}/sbin/start-master.sh; ${SPARK_HOME}/sbin/start-worker.sh -c ${CORES_PER_WORKER} -m 16G ${MASTER}

# start jupyter with pyspark
${SPARK_HOME}/bin/pyspark --master ${MASTER} \
--driver-memory 8G \
--executor-memory 8G \
--conf spark.python.worker.reuse=True

# BROWSE to localhost:8888 to view/run notebooks

# stop spark standalone cluster
${SPARK_HOME}/sbin/stop-worker.sh; ${SPARK_HOME}/sbin/stop-master.sh
```

## Triton Inference Server

The example notebooks also demonstrate integration with [Triton Inference Server](https://developer.nvidia.com/nvidia-triton-inference-server), an open-source, GPU-accelerated serving solution for DL.

**Note**: Some examples may require special configuration of server as highlighted in the notebooks.

**Note**: for demonstration purposes, the Triton Inference Server integrations just launch the server in a docker container on the local host, so you will need to [install docker](https://docs.docker.com/engine/install/) on your local host.  Most real-world deployments will likely be hosted on remote machines.
