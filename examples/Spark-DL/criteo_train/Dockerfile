#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

ARG IMAGE=nvcr.io/nvidia/tensorflow:21.03-tf2-py3
FROM ${IMAGE}
ENV CUDA_SHORT_VERSION=11.2

SHELL ["/bin/bash", "-c"]
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/lib:/repos/dist/lib

ENV DEBIAN_FRONTEND=noninteractive

ARG RELEASE=true
ARG RMM_VER=v21.10.00
ARG CUDF_VER=v21.10.00
ARG NVTAB_VER=v0.6.0
ARG SM="60;61;70;75;80"

ENV CUDA_HOME=/usr/local/cuda
ENV CUDA_PATH=$CUDA_HOME
ENV CUDA_CUDA_LIBRARY=${CUDA_HOME}/lib64/stubs
ENV PATH=${CUDA_HOME}/lib64/:${PATH}:${CUDA_HOME}/bin

# Build env variables for rmm
ENV INSTALL_PREFIX=/usr

RUN apt update -y --fix-missing && \
    apt upgrade -y && \
      apt install -y --no-install-recommends software-properties-common && \
      apt update -y --fix-missing

RUN apt install -y --no-install-recommends \
      git \
      libboost-all-dev \
      python3.8-dev \
      build-essential \
      autoconf \
      bison \
      flex \
      libboost-filesystem-dev \
      libboost-system-dev \
      libboost-regex-dev \
      libjemalloc-dev \
      wget \
      libssl-dev \
      protobuf-compiler \
      clang-format \
      aptitude \
      numactl \
      libnuma-dev \
      libaio-dev \
      libibverbs-dev \
      libtool && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* 
    #update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1 && \
    #wget https://bootstrap.pypa.io/get-pip.py && \
    #python get-pip.py

# Install cmake
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null && \
    apt-add-repository 'deb https://apt.kitware.com/ubuntu/ focal main' && \
    apt-get update && \
    apt-get install -y cmake

# Install arrow from source
ENV ARROW_HOME=/usr/local
RUN git clone --branch apache-arrow-4.0.1 --recurse-submodules https://github.com/apache/arrow.git build-env && \
    pushd build-env && \
      export PARQUET_TEST_DATA="${PWD}/cpp/submodules/parquet-testing/data" && \
      export ARROW_TEST_DATA="${PWD}/testing/data" && \
      pip install -r python/requirements-build.txt && \
      mkdir cpp/release && \
      pushd cpp/release && \
        cmake -DCMAKE_INSTALL_PREFIX=${ARROW_HOME} \
              -DCMAKE_INSTALL_LIBDIR=lib \
              -DCMAKE_LIBRARY_PATH=${CUDA_CUDA_LIBRARY} \
              -DARROW_FLIGHT=ON \
              -DARROW_GANDIVA=OFF \
              -DARROW_ORC=ON \
              -DARROW_WITH_BZ2=ON \
              -DARROW_WITH_ZLIB=ON \
              -DARROW_WITH_ZSTD=ON \
              -DARROW_WITH_LZ4=ON \
              -DARROW_WITH_SNAPPY=ON \
              -DARROW_WITH_BROTLI=ON \
              -DARROW_PARQUET=ON \
              -DARROW_PYTHON=ON \
              -DARROW_PLASMA=ON \
              -DARROW_BUILD_TESTS=ON \
              -DARROW_CUDA=ON \
              -DARROW_DATASET=ON \
              .. && \
        make -j$(nproc) && \
        make install && \
      popd && \
      pushd python && \
        export PYARROW_WITH_PARQUET=ON && \
        export PYARROW_WITH_CUDA=ON && \
        export PYARROW_WITH_ORC=ON && \
        export PYARROW_WITH_DATASET=ON && \
        python setup.py build_ext --build-type=release bdist_wheel && \
        pip install dist/*.whl && \
      popd && \
    popd && \
    rm -rf build-env


# Install rmm from source
RUN git clone https://github.com/rapidsai/rmm.git build-env && cd build-env/ && \
    if [ "$RELEASE" == "true" ] && [ ${RMM_VER} != "vnightly" ] ; then git fetch --all --tags && git checkout tags/${RMM_VER}; else git checkout main; fi; \
    sed -i '/11.2/ a "11.4": "11.x",' python/setup.py && \
    cd ..; \
    pushd build-env && \
    ./build.sh librmm && \
    pip install python/. && \
    popd && \
    rm -rf build-env



# Build env for CUDF build
RUN git clone https://github.com/rapidsai/cudf.git build-env && cd build-env/ && \
    if [ "$RELEASE" == "true" ] && [ ${CUDF_VER} != "vnightly" ] ; then git fetch --all --tags && git checkout tags/${CUDF_VER}; else git checkout main; fi; \
    git submodule update --init --recursive && \
    cd .. && \
    pushd build-env && \
      export CUDF_HOME=${PWD} && \
      export CUDF_ROOT=${PWD}/cpp/build/ && \
      export CMAKE_LIBRARY_PATH=${CUDA_CUDA_LIBRARY} && \
      ./build.sh libcudf cudf dask_cudf && \
      protoc -I=python/cudf/cudf/utils/metadata --python_out=/usr/local/lib/python3.8/dist-packages/cudf/utils/metadata python/cudf/cudf/utils/metadata/orc_column_statistics.proto && \
    popd && \
    rm -rf build-env

RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        vim gdb git wget unzip tar python3.8-dev \
        zlib1g-dev lsb-release clang-format libboost-all-dev \
        openssl curl zip\
       	slapd && \
    rm -rf /var/lib/apt/lists/*

ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION='python'


RUN pip install pandas sklearn ortools nvtx-plugins pydot && \
    pip cache purge

# tf-nightly for performance test
# more details: https://github.com/tensorflow/tensorflow/issues/44194
RUN pip uninstall tensorflow -y; pip install tf-nightly==2.7.0.dev20210722
RUN pip uninstall keras-nightly -y; pip install keras-nightly==2.7.0.dev2021072200


RUN mkdir -p /usr/local/nvidia/lib64 && \
    ln -s /usr/local/cuda/lib64/libcusolver.so /usr/local/nvidia/lib64/libcusolver.so.10

RUN pip install pybind11
SHELL ["/bin/bash", "-c"]

# prepare nccl
RUN apt remove -y libnccl2 libnccl-dev
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin \
    && mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600 \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub \
    && add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /" \
    && apt-get update \
    && apt install libnccl2=2.8.4-1+cuda11.2 libnccl-dev=2.8.4-1+cuda11.2

# install Horovod
RUN pip uninstall horovod -y
RUN HOROVOD_WITH_MPI=1 HOROVOD_WITH_TENSORFLOW=1 HOROVOD_GPU_OPERATIONS=NCCL \
    pip install horovod[spark] --no-cache-dir

# Install NVTabular
RUN git clone https://github.com/NVIDIA/NVTabular.git /nvtabular/ && \
    cd /nvtabular/; if [ "$RELEASE" == "true" ] && [ ${NVTAB_VER} != "vnightly" ] ; then git fetch --all --tags && git checkout tags/${NVTAB_VER}; else git checkout main; fi; \
    python setup.py develop --user;


RUN pip install pynvml pytest graphviz sklearn scipy matplotlib
RUN pip install nvidia-pyindex; pip install tritonclient[all] grpcio-channelz
RUN pip install nvtx mpi4py==3.0.3 cupy-cuda112 cachetools typing_extensions fastavro

RUN apt-get update; apt-get install -y graphviz

RUN pip uninstall numpy -y; pip install numpy
RUN pip install dask==2021.04.1 distributed==2021.04.1 dask-cuda
RUN pip install dask[dataframe]==2021.04.1
RUN pip uninstall pandas -y; pip install pandas==1.1.5
RUN echo $(du -h --max-depth=1 /)


# install spark-3.1.2-bin-hadoop3.2
RUN wget \
    https://mirror-hk.koddos.net/apache/spark/spark-3.1.2/spark-3.1.2-bin-hadoop3.2.tgz \
    && tar -xzf spark-3.1.2-bin-hadoop3.2.tgz -C /opt/ \
    && rm spark-3.1.2-bin-hadoop3.2.tgz

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get install openjdk-8-jdk openjdk-8-jre lsb-release -y --allow-downgrades --allow-change-held-packages --no-install-recommends
ENV JAVA_HOME /usr/lib/jvm/java-1.8.0-openjdk-amd64
ENV PATH $PATH:/usr/lib/jvm/java-1.8.0-openjdk-amd64/jre/bin:/usr/lib/jvm/java-1.8.0-openjdk-amd64/bin

# add spark env to conf
ADD spark-env.sh /opt/spark-3.1.2-bin-hadoop3.2/conf/

ADD start-spark.sh /workspace/
ADD submit.sh /workspace/
ADD criteo_keras.py /workspace/

HEALTHCHECK NONE
ENTRYPOINT []
CMD ["/bin/bash"]
