# Spark DL Inference on AWS EMR

The instructions assume you already have a AWS EMR account. 
**Note**: fields in \<brackets\> require user inputs.

## Setup

#### Setup AWS CLI

1. Install the [AWS CLI](https://docs.aws.amazon.com/emr/latest/EMR-on-EKS-DevelopmentGuide/setting-up-cli.html). 

2. Initialize the CLI via `aws configure`. You may need to create access keys by following [Authenticating using IAM user credentials](https://docs.aws.amazon.com/cli/latest/userguide/cli-authentication-user.html). You can find your default region name (e.g. Ohio) on the right of the top navigation bar. Clicking the region name will show the region code (e.g. us-east-2 for Ohio). 

    ```shell
    aws configure
    AWS Access Key ID [None]: <your_access_key>
    AWS Secret Access Key [None]: <your_secret_access_key>
    Default region name [None]: <region-code>
    Default output format [None]: json
    ```

#### Copy Files to S3

3. Create an S3 bucket if you don't already have one.
    ```shell
    export S3_BUCKET=<your_s3_bucket_name>
    aws s3 mb s3://${S3_BUCKET}
    ```

4. Upload the initialization script to S3.
    ```shell
    aws s3 cp spark_rapids_ml.zip s3://${S3_BUCKET}/spark_rapids_ml.zip
    cd ../../notebooks/aws-emr
    aws s3 cp init-bootstrap-action.sh s3://${S3_BUCKET}/init-bootstrap-action.sh
    ```
- Print out available subnets in CLI then pick a SubnetId (e.g. subnet-0744566f of AvailabilityZone us-east-2a).
    ```shell
    aws ec2 describe-subnets
    export SUBNET_ID=<your_SubnetId>
    ```

- Create a cluster with at least two single-gpu workers. You will obtain a ClusterId in terminal. Noted three GPU nodes are requested here, because EMR cherry picks one node (either CORE or TASK) to run JupyterLab service for notebooks and will not use the node for compute.
    ```shell
    export CLUSTER_NAME="spark_rapids_ml"
    export CUR_DIR=$(pwd)

    aws emr create-cluster \
    --name ${CLUSTER_NAME} \
    --release-label emr-7.3.0 \
    --ebs-root-volume-size=32 \
    --applications Name=Hadoop Name=Livy Name=Spark Name=JupyterEnterpriseGateway \
    --service-role EMR_DefaultRole \
    --log-uri s3://${S3_BUCKET}/logs \
    --ec2-attributes SubnetId=${SUBNET_ID},InstanceProfile=EMR_EC2_DefaultRole \
    --instance-groups InstanceGroupType=MASTER,InstanceCount=1,InstanceType=m4.2xlarge \
                    InstanceGroupType=CORE,InstanceCount=3,InstanceType=g4dn.2xlarge \
    --configurations file://${CUR_DIR}/init-configurations.json \
    --bootstrap-actions Name='Spark Rapids ML Bootstrap action',Path=s3://${S3_BUCKET}/init-bootstrap-action.sh
    ```

- In the [AWS EMR console](https://console.aws.amazon.com/emr/), click "Clusters", you can find the cluster id of the created cluster. Wait until all the instances have the Status turned to "Running".
- In the [AWS EMR console](https://console.aws.amazon.com/emr/), click "Workspace(Notebooks)", then create a workspace. Wait until the status becomes ready and a JupyterLab webpage will pop up. 

- Enter the created workspace. Click the "Cluster" button (usually the top second button of the left navigation bar). Attach the workspace to the newly created cluster through cluster id.

- Use the default notebook or create/upload a new notebook. Set the notebook kernel to "PySpark".  

- Add the following to a new cell at the beginning of the notebook. Replace "s3://path/to/spark\_rapids\_ml.zip" with the actual s3 path.  
    ```python
    %%configure -f
    {
        "conf":{
            "spark.submit.pyFiles": "s3://path/to/spark_rapids_ml.zip"
        }
    }
  
  ```
- Run the notebook cells.  
  **Note**: these settings are for demonstration purposes only.  Additional tuning may be required for optimal performance.
