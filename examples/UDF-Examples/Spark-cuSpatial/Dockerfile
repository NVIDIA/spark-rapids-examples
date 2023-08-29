#!/bin/bash
#
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

ARG CUDA_VER=11.2.2
FROM nvidia/cuda:${CUDA_VER}-devel-ubuntu18.04

RUN apt-key adv --fetch-keys https://developer.download.nvidia.cn/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
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

RUN wget --quiet \
    https://github.com/Kitware/CMake/releases/download/v3.26.4/cmake-3.26.4-linux-x86_64.tar.gz \
    && tar -xzf cmake-3.21.3-linux-x86_64.tar.gz \
    && rm -rf cmake-3.21.3-linux-x86_64.tar.gz

ENV PATH="/cmake-3.21.3-linux-x86_64/bin:${PATH}"