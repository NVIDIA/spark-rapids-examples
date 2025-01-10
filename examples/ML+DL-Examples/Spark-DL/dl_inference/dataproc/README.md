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

4.  Specify the local path to the notebook(s) and copy to the GCS bucket.
    As an example for a torch notebook:
    ```shell
    export SPARK_DL_HOME=${GCS_BUCKET}/spark-dl
    gcloud storage cp </path/to/notebook_name_torch.ipynb> gs://${SPARK_DL_HOME}/notebooks/
    ```
    All notebooks under `gs://${SPARK_DL_HOME}/notebooks/` will be copied to the master node during initialization.

#### Start cluster and run

5. Specify the framework to use ('torch' or 'tf'), which will determine what libraries to install on the cluster. For example:
    ```shell
    export FRAMEWORK=torch
    ```
    Run the cluster startup script. The script will also retrieve and use the [spark-rapids initialization script](https://github.com/GoogleCloudDataproc/initialization-actions/blob/master/spark-rapids/spark-rapids.sh) to setup GPU resources.
    ```shell
    cd setup
    chmod +x start_cluster.sh
    ./start_cluster.sh
    ```
    The script creates a 4 node GPU cluster by default.
    The cluster name will default to `${USER}-spark-dl-inference-${FRAMEWORK}`.

7. Browse to the Jupyter web UI:
    - Go to `Dataproc` > `Clusters` > `(Cluster Name)` > `Web Interfaces` > `Jupyter/Lab`
    
    Or, get the link by running this command (link is under httpPorts > Jupyter/Lab):
    ```shell
    gcloud dataproc clusters describe ${CLUSTER_NAME} --region=${COMPUTE_REGION}
    ```

8. Open and run the notebook interactively with the Python 3 kernel.  
The notebooks can be found under `Local Disk/spark-dl-notebooks` on the master node.  