#!/bin/bash
#
# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
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
#

ARG CUDA_VER=11.8.0
FROM nvidia/cuda:${CUDA_VER}-devel-ubuntu20.04
# Please do not update the BRANCH_VER version
ARG BRANCH_VER=24.04

RUN apt-get update
RUN apt-get install -y wget ninja-build git

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN wget --quiet \
    https://repo.anaconda.com/miniconda/Miniconda3-py38_4.10.3-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-py38_4.10.3-Linux-x86_64.sh -b \
    && rm -f Miniconda3-py38_4.10.3-Linux-x86_64.sh \
    && conda init

SHELL ["conda", "run", "--no-capture-output", "-n", "base", "/bin/bash", "-c"]
RUN echo $PATH
RUN echo $CONDA_PREFIX
RUN conda --version

RUN conda install -c conda-forge openjdk=8 maven=3.8.1 -y

# install cuDF dependency.
RUN conda install -c rapidsai-nightly -c nvidia -c conda-forge cudf=${BRANCH_VER} python=3.8 -y

RUN wget --quiet \
    https://github.com/Kitware/CMake/releases/download/v3.26.4/cmake-3.26.4-linux-x86_64.tar.gz \
    && tar -xzf cmake-3.21.3-linux-x86_64.tar.gz \
    && rm -rf cmake-3.21.3-linux-x86_64.tar.gz

ENV PATH="/cmake-3.21.3-linux-x86_64/bin:${PATH}"

RUN git clone -b branch-${BRANCH_VER} https://github.com/rapidsai/raft.git

ENV RAFT_PATH=/raft

#use main branch to download release jars
RUN git clone -b main https://github.com/NVIDIA/spark-rapids-ml.git \
    && cd spark-rapids-ml \
    && mvn clean
RUN cd /spark-rapids-ml \
    && mvn install -DskipTests

ADD scala /workspace/scala
ADD pom.xml /workspace/

RUN cd /workspace/ \
    && mvn clean package

# install spark-3.1.2-bin-hadoop3.2
RUN wget --quiet \
    https://archive.apache.org/dist/spark/spark-3.1.2/spark-3.1.2-bin-hadoop3.2.tgz \
    && tar -xzf spark-3.1.2-bin-hadoop3.2.tgz -C /opt/ \
    && rm spark-3.1.2-bin-hadoop3.2.tgz
ENV SPARK_HOME=/opt/spark-3.1.2-bin-hadoop3.2
# add spark env to conf
ADD spark-env.sh /opt/spark-3.1.2-bin-hadoop3.2/conf/
ADD start-spark.sh /workspace/
ADD spark-submit.sh /workspace/

WORKDIR /workspace
