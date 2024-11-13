# Spark DL Inference on Dataproc

Distributed deep learning inference using the PySpark [predict_batch_udf](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.functions.predict_batch_udf.html#pyspark.ml.functions.predict_batch_udf) function on GCP Dataproc.  
The examples also demonstrate integration with [Triton Inference Server](https://developer.nvidia.com/nvidia-triton-inference-server), an open-source, GPU-accelerated serving solution for DL. 

## Setup

1. Install the latest [gcloud-cli](https://cloud.google.com/sdk/docs/install) and configure for your workspace.

2. `cd` into the [setup directory](setup).

3. Configure the following settings:
    ```shell
    export PROJECT=<your_project>
    export DATAPROC_REGION=<your_dataproc_region>
    export COMPUTE_REGION=<your_compute_region>
    export COMPUTE_ZONE=<your_compute_zone>

    gcloud config set project ${PROJECT}
    gcloud config set dataproc/region ${DATAPROC_REGION}
    gcloud config set compute/region ${COMPUTE_REGION}
    gcloud config set compute/zone ${COMPUTE_ZONE}
    ```

4. Create a GCS bucket if you don't already have one:
    ```shell
    export GCS_BUCKET=<your_gcs_bucket_name>

    gcloud storage buckets create gs://${GCS_BUCKET}
    ```

5. Run the setup script, which will copy files to your GCS bucket: 
    ```shell
    export SPARK_DL_HOME=${GCS_BUCKET}/spark-dl
    
    cd setup
    chmod +x setup.sh
    ./setup.sh
    ```

6. Launch the cluster (default to 2 node GPU cluster):
    ```shell
    chmod +x start_cluster.sh
    ./start_cluster.sh
    ```
    The default cluster name will be `"${USER}-spark-dl-gpu"`.

7. Attach to the Jupyter web UI. You can get the link by running this command:
    ```shell
    export CLUSTER_NAME=${USER}-spark-dl-gpu
    gcloud dataproc clusters describe ${CLUSTER_NAME} --region=${COMPUTE_REGION}
    ```

    OR, you can find the links on the GCP web UI:
    - Go to `Dataproc` > `Clusters` > `<cluster_name>` > `Web Interfaces` > `Jupyter`

8. Open and run the notebook interactively. 
The init script copies the notebook to `Local disk` > `notebooks` > `conditional_generation.ipynb` on the master node.

9. To cleanup, you can delete the cluster:
    ```shell
    gcloud dataproc clusters delete ${CLUSTER_NAME} --region=${COMPUTE_REGION}
    ```
    and the bucket:
    ```shell
    gcloud 
    ```