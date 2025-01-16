#!/bin/bash -ex
# Copyright (c) 2024, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -o pipefail

cluster_type=${1:-gpu}

# configure arguments
if [[ -z ${SUBNET_ID} ]]; then
    echo "Please export SUBNET_ID per README.md"
    exit 1
fi

if [[ -z ${BENCHMARK_HOME} ]]; then
    echo "Please export BENCHMARK_HOME per README.md"
    exit 1
fi

if [[ -z ${KEYPAIR} ]]; then
    echo "Please export KEYPAIR per README.md"
    exit 1
fi

cluster_name=spark-rapids-ml-${cluster_type}
cur_dir=$(pwd)

if [[ ${cluster_type} == "gpu" ]]; then
    core_type=g5.2xlarge
    config_json="file://${cur_dir}/../../../notebooks/aws-emr/init-configurations.json"
    bootstrap_actions="--bootstrap-actions Name='Spark Rapids ML Bootstrap action',Path=s3://${BENCHMARK_HOME}/init-bootstrap-action.sh"
elif [[ ${cluster_type} == "cpu" ]]; then
    core_type=m6gd.2xlarge
    config_json="file://${cur_dir}/cpu-init-configurations.json"
    bootstrap_actions=""
else
    echo "unknown cluster type ${cluster_type}"
    echo "usage: ./${script_name} cpu|gpu"
    exit 1
fi

start_cmd="aws emr create-cluster \
--name ${cluster_name} \
--release-label emr-7.3.0 \
--applications Name=Hadoop Name=Spark \
--service-role EMR_DefaultRole \
--log-uri s3://${BENCHMARK_HOME}/logs \
--ec2-attributes KeyName=$(basename ${KEYPAIR} | sed -e 's/\.pem//g' ),SubnetId=${SUBNET_ID},InstanceProfile=EMR_EC2_DefaultRole \
--ebs-root-volume-size=32 \
--instance-groups InstanceGroupType=MASTER,InstanceCount=1,InstanceType=m4.2xlarge \
                  InstanceGroupType=CORE,InstanceCount=3,InstanceType=${core_type} \
--configurations ${config_json} $bootstrap_actions
"

CLUSTER_ID=$( eval ${start_cmd} | tee /dev/tty | grep "ClusterId" | grep -o 'j-[0-9|A-Z]*')
aws emr put-auto-termination-policy --cluster-id ${CLUSTER_ID} --auto-termination-policy IdleTimeout=1800
echo "waiting for cluster ${CLUSTER_ID} to start ... " 1>&2

aws emr wait cluster-running --cluster-id $CLUSTER_ID

echo "cluster started." 1>&2
echo $CLUSTER_ID
