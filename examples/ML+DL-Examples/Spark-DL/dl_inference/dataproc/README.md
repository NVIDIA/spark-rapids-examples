# Spark DL Inference on Dataproc

## Setup

**Note**: fields in \<brackets\> require user inputs.

#### Setup GCloud CLI

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

#### Copy files to GCS

3. Create a GCS bucket if you don't already have one:
    ```shell
    export GCS_BUCKET=<your_gcs_bucket_name>

    gcloud storage buckets create gs://${GCS_BUCKET} 
    ```

4.  Specify the local paths to the notebook and init script (_torch or _tf). 
    As an example for a PyTorch notebook:
    ```shell
    export NOTEBOOK_SRC=</path/to/notebook_name_torch.ipynb>
    export INIT_SRC=</path/to/setup/init_spark_dl_torch.sh>
    ```
    
5. Run the commands below to copy the files to the GCS bucket:
    ```shell
    export SPARK_DL_HOME=${GCS_BUCKET}/spark-dl
    gcloud storage cp ${INIT_SRC} gs://${SPARK_DL_HOME}/init/
    gcloud storage cp ${NOTEBOOK_SRC} gs://${SPARK_DL_HOME}/notebooks/
    ```

#### Start cluster and run

6. Launch the cluster:
    ```shell
    cd setup
    chmod +x start_cluster.sh
    ./start_cluster.sh
    ```
    The script creates a 4 node GPU cluster by default.
    The cluster name will default to `${USER}-spark-dl-inference-torch` or `${USER}-spark-dl-inference-tf` based on the init script used.

7. Browse to the Jupyter web UI. You can get the link by running this command:
    ```shell
    gcloud dataproc clusters describe ${CLUSTER_NAME} --region=${COMPUTE_REGION}
    ```

    OR you can find the links on the GCP web UI:
    - Go to `Dataproc` > `Clusters` > `(Cluster Name)` > `Web Interfaces` > `Jupyter`

8. Open and run the notebook interactively with the Python 3 kernel.  
The init script copies the notebook to `Local Disk/spark-dl-notebooks` on the master node.  