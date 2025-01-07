# Spark DL Inference on Dataproc

## Setup

1. Install the latest [gcloud-cli](https://cloud.google.com/sdk/docs/install) and configure for your workspace.

2. Configure the following settings:
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

3. Create a GCS bucket if you don't already have one:
    ```shell
    export GCS_BUCKET=<your_gcs_bucket_name>

    gcloud storage buckets create gs://${GCS_BUCKET}
    ```

4.  Specify the paths below to the notebook and init script (_torch or _tf). As an example for a PyTorch notebook:
    ```shell
    export NOTEBOOK_SRC=/path/to/notebook_torch.ipynb
    export INIT_SRC=/path/to/setup/init_spark_dl_torch.sh
    ```

5. Run the commands below to copy the files to the GCS bucket:
    ```shell
    export SPARK_DL_HOME=${GCS_BUCKET}/spark-dl
    gcloud storage cp ${INIT_SRC} gs://${SPARK_DL_HOME}/init_spark_dl_torch.sh
    gcloud storage cp ${NOTEBOOK_SRC} gs://${SPARK_DL_HOME}/conditional_generation.ipynb
    ```

6. Launch the cluster (defaults to 4 node GPU cluster):
    ```shell
    export CLUSTER_NAME=${USER}-spark-dl-inference-torch
    cd setup
    chmod +x start_cluster.sh
    ./start_cluster.sh
    ```

7. Browse to the Jupyter web UI. You can get the link by running this command:
    ```shell
    gcloud dataproc clusters describe ${CLUSTER_NAME} --region=${COMPUTE_REGION}
    ```

    OR you can find the links on the GCP web UI:
    - Go to `Dataproc` > `Clusters` > `(Cluster Name)` > `Web Interfaces` > `Jupyter`

8. Open and run the notebook interactively with the Python 3 kernel.  
The init script copies the notebook to `Local Disk/notebooks` on the master node.  