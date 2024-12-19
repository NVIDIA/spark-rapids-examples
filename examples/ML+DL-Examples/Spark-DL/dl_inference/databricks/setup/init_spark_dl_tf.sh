#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

set -x

# install requirements
sudo /databricks/python3/bin/pip3 install --upgrade pip
sudo /databricks/python3/bin/pip3 install datasets==3.0.1
sudo /databricks/python3/bin/pip3 install nvidia-pytriton==0.5.11 protobuf==3.20.2

set +x