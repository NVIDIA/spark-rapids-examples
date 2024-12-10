import argparse
import subprocess
import json
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError

def upload_file_to_s3(file_name, bucket_name, object_name=None):
    s3 = boto3.client('s3')

    # If no object name is specified, use the file name
    if object_name is None:
        object_name = file_name

    try:
        s3.upload_file(file_name, bucket_name, object_name)
        print(f"File '{file_name}' uploaded successfully to bucket '{bucket_name}' as '{object_name}'")
    except FileNotFoundError:
        print(f"Error: The file {file_name} was not found.")
    except NoCredentialsError:
        print("Error: AWS credentials not found.")
    except PartialCredentialsError:
        print("Error: Incomplete AWS credentials.")
    except Exception as e:
        print(f"An error occurred: {e}")

g4dn_instance_map = {
    "g4dn.xlarge": 4,
    "g4dn.2xlarge": 8,
    "g4dn.4xlarge": 16,
    "g4dn.12xlarge": 48,
    "g4dn.16xlarge": 64
}

def create_emr_cluster(release_label, key_name, service_role, subnet_id, az, instance_profile, worker_instance, s3_bucket_name):
    try:
        conf_json_fn = None
        bootstrap_fn = None
        if "emr-7" in release_label:
            conf_json_fn="config-emr7.json"
            bootstrap_fn="cgroup-bootstrap-action-emr7.sh"
        else:
            conf_json_fn="config-emr6.json"
            bootstrap_fn="cgroup-bootstrap-action-emr6.sh"
        # Replace the fields in the json
        exec_cores = g4dn_instance_map.get(worker_instance)

        print("Config Json" + conf_json_fn)
        with open(conf_json_fn, 'r') as file:
            data = json.load(file)
        json_string = json.dumps(data)

        # Replace the placeholder with the actual variable
        json_string = json_string.replace("${task_gpu_amount}", str(1/exec_cores))
        json_string = json_string.replace("${executor_cores}", str(exec_cores))
        updated_data = json.loads(json_string)

        print(json.dumps(updated_data, indent=4))
        with open(conf_json_fn, "w") as file:
            file.write(json_string)
        upload_file_to_s3(bootstrap_fn, s3_bucket_name, bootstrap_fn)


        command = [
            "aws", "emr", "create-cluster",
            "--release-label", release_label,
            "--applications", "Name=Hadoop", "Name=Spark", "Name=Livy", "Name=JupyterEnterpriseGateway",
            "--service-role", service_role,
            "--ec2-attributes", f"AvailabilityZone={az},InstanceProfile={instance_profile}",
            "--instance-groups",
            "InstanceGroupType=MASTER,InstanceCount=1,InstanceType=m4.4xlarge",
            f"InstanceGroupType=CORE,InstanceCount=1,InstanceType={worker_instance}",
            "--configurations", f"file://{conf_json_fn}",
            "--bootstrap-actions", f"Name='Setup cgroups bootstrap',Path=s3://{s3_bucket_name}/{bootstrap_fn}"
        ]

        result = subprocess.run(command, check=True, text=True, capture_output=True)

        print("Cluster created successfully!")
        print(result.stdout)

    except subprocess.CalledProcessError as e:
        print("Error creating EMR cluster:", e.stderr)


parser = argparse.ArgumentParser(description="A script that takes command-line arguments.")

# Define arguments
parser.add_argument("-r", "--release_label", type=str, default="emr-7.1.0",  help="EMR Release Label, emr-7.1.0 for example")
parser.add_argument("-k", "--key_name", type=str, required=True, help="Access Key Name")
parser.add_argument("-s", "--service_role", type=str, required=True, help="AWS EMR service Role")
parser.add_argument("-n", "--subnet", type=str, required=True, help="Subnet ID")
parser.add_argument("-z", "--availability_zone", type=str, default="us-west-2b", help="Availability Zone")
parser.add_argument("-i", "--instance_profile", type=str, required=True, help="Instance Profile")
parser.add_argument("-w", "--worker_instance", type=str, default="g4dn.2xlarge",  help="Worker Instance g4dn.xxxx")
parser.add_argument("-b", "--s3_bucket_name", type=str, required=True, help="S3 Bucket Name to store the bootstrap and config info")

args = parser.parse_args()

release_label = args.release_label
key_name = args.key_name
service_role = args.service_role
subnet_id = args.subnet
az = args.availability_zone
instance_profile = args.instance_profile
worker_instance = args.worker_instance
s3_bucket_name = args.s3_bucket_name

create_emr_cluster(release_label, key_name, service_role, subnet_id, az, instance_profile, worker_instance, s3_bucket_name)
