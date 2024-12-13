# Spark DL Inference Using External Frameworks

Example notebooks for the [predict_batch_udf](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.functions.predict_batch_udf.html#pyspark.ml.functions.predict_batch_udf) function introduced in Spark 3.4.

The example notebooks also demonstrate integration with [Triton Inference Server](https://developer.nvidia.com/nvidia-triton-inference-server), an open-source, GPU-accelerated serving solution for DL.

## Contents:
- [Overview](#overview)
- [Running Locally](#running-the-notebooks)
- [Running on Cloud](#running-on-cloud-environments)

## Overview

This directory contains notebooks for each DL framework (based on their own published examples).  The goal is to demonstrate how models trained and saved on single-node machines can be easily used for parallel inferencing on Spark clusters.

For example, a basic model trained in TensorFlow and saved on disk as "mnist_model" can be used in Spark as follows:
```
import numpy as np
from pyspark.sql.functions import predict_batch_udf
from pyspark.sql.types import ArrayType, FloatType

def predict_batch_fn():
    import tensorflow as tf
    model = tf.keras.models.load_model("/path/to/mnist_model")
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

In this simple case, the `predict_batch_fn` will use TensorFlow APIs to load the model and return a simple `predict` function which operates on numpy arrays.  The `predict_batch_udf` will automatically convert the Spark DataFrame columns to the expected numpy inputs.

All notebooks have been saved with sample outputs for quick browsing.  
Here is a full list of the notebooks with their published example links:

|   | Category  | Notebook Name | Description | Link
| ------------- | ------------- | ------------- | ------------- | ------------- 
| 1 | PyTorch | Image Classification | Training a model to predict clothing categories in FashionMNIST, including accelerated inference with Torch-TensorRT. | [Link](https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html)
| 2 | PyTorch | Housing Regression | Training a model to predict housing prices in the California Housing Dataset, including accelerated inference with Torch-TensorRT. | [Link](https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-create-a-neural-network-for-regression-with-pytorch.md)
| 3 | Tensorflow | Image Classification | Training a model to predict hand-written digits in MNIST. | [Link](https://github.com/tensorflow/docs/blob/master/site/en/tutorials/keras/save_and_load.ipynb)
| 4 | Tensorflow | Keras Preprocessing | Training a model with preprocessing layers to predict likelihood of pet adoption in the PetFinder mini dataset. | [Link](https://github.com/tensorflow/docs/blob/master/site/en/tutorials/structured_data/preprocessing_layers.ipynb)
| 5 | Tensorflow | Keras Resnet50 | Training ResNet-50 to perform flower recognition from flower images. | [Link](https://docs.databricks.com/en/_extras/notebooks/source/deep-learning/keras-metadata.html)
| 6 | Tensorflow | Text Classification | Training a model to perform sentiment analysis on the IMDB dataset. | [Link](https://github.com/tensorflow/docs/blob/master/site/en/tutorials/keras/text_classification.ipynb)
| 7+8 | HuggingFace | Conditional Generation | Sentence translation using the T5 text-to-text transformer, with notebooks demoing both Torch and Tensorflow. | [Link](https://huggingface.co/docs/transformers/model_doc/t5#t5) 
| 9+10 | HuggingFace | Pipelines | Sentiment analysis using Huggingface pipelines, with notebooks demoing both Torch and Tensorflow. | [Link](https://huggingface.co/docs/transformers/quicktour#pipeline-usage)
| 11 | HuggingFace | Sentence Transformers | Sentence embeddings using SentenceTransformers in Torch. | [Link](https://huggingface.co/sentence-transformers)

## Running the Notebooks

If you want to run the notebooks yourself, please follow these instructions.

**Notes**: 
- The notebooks require a GPU environment for the executors.  
- Please create separate environments for PyTorch and Tensorflow examples as specified below. This will avoid conflicts between the CUDA libraries bundled with their respective versions. The Huggingface examples will have a `_torch` or `_tf` suffix to specify the environment used.
- The PyTorch notebooks include model compilation and accelerated inference with TensorRT. While not included in the notebooks, Tensorflow also supports [integration with TensorRT](https://docs.nvidia.com/deeplearning/frameworks/tf-trt-user-guide/index.html), but as of writing it is not supported in TF==2.17.0. 

#### Create environment

**For PyTorch:**
```
conda create -n spark-dl-torch python=3.11
conda activate spark-dl-torch
pip install -r torch_requirements.txt
```
**For TensorFlow:**
```
conda create -n spark-dl-tf python=3.11
conda activate spark-dl-tf
pip install -r tf_requirements.txt
```

#### Launch Jupyter + Spark

For demonstration, these examples just use a local Standalone cluster with a single executor, but you may run them on any distributed Spark cluster.
```
# setup environment variables
export SPARK_HOME=/path/to/spark
export MASTER=spark://$(hostname):7077
export SPARK_WORKER_INSTANCES=1
export SPARK_WORKER_OPTS="-Dspark.worker.resource.gpu.amount=1 -Dspark.worker.resource.gpu.discoveryScript=$SPARK_HOME/examples/src/main/scripts/getGpusResources.sh"
export CORES_PER_WORKER=8
export PYSPARK_DRIVER_PYTHON=jupyter
export PYSPARK_DRIVER_PYTHON_OPTS='lab'

# start spark standalone cluster
${SPARK_HOME}/sbin/start-master.sh; ${SPARK_HOME}/sbin/start-worker.sh -c ${CORES_PER_WORKER} -m 16G ${MASTER}

# start jupyter with pyspark
${SPARK_HOME}/bin/pyspark --master ${MASTER} \
--driver-memory 8G \
--executor-memory 8G \
--conf spark.executor.resource.gpu.amount=1 \
--conf spark.task.resource.gpu.amount=0.125 \
--conf spark.executor.cores=8 \
--conf spark.executorEnv.LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${CONDA_PREFIX}/lib/python3.11/site-packages/nvidia_pytriton.libs:$LD_LIBRARY_PATH"

# BROWSE to localhost:8888 to view/run notebooks

# stop spark standalone cluster
${SPARK_HOME}/sbin/stop-worker.sh; ${SPARK_HOME}/sbin/stop-master.sh
```

**Troubleshooting:** If you encounter issues starting the Triton server, you may need to link your libstdc++ file to the conda environment, e.g.:
```shell
ln -sf /usr/lib/x86_64-linux-gnu/libstdc++.so.6 ${CONDA_PREFIX}/lib/libstdc++.so.6
```

## Running on cloud environments

We also provide instructions to run the notebooks on CSP Spark environments.  
See the instructions for [Databricks](databricks/README.md) and [GCP Dataproc](dataproc/README.md) respectively. 
