ARG CUDA_VER=11.2.2
FROM nvidia/cuda:${CUDA_VER}-devel-ubuntu20.04

RUN apt-get update
RUN apt-get install -y wget ninja-build git

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN wget --quiet \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh

RUN conda --version
   

RUN conda install openjdk=8 maven=3.8.1 -y

RUN wget --quiet \
    https://github.com/Kitware/CMake/releases/download/v3.21.3/cmake-3.21.3-linux-x86_64.tar.gz \
    && tar -xzf cmake-3.21.3-linux-x86_64.tar.gz \
    && rm -rf cmake-3.21.3-linux-x86_64.tar.gz \

ENV PATH="/cmake-3.21.3-linux-x86_64/bin:${PATH}"

RUN git clone --recurse-submodules https://github.com/rapidsai/rmm.git \
    && cd rmm \
    && mkdir build \
    && cd build \
    && cmake .. -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX \
    && make -j \
    && make install

RUN git clone https://github.com/rapidsai/raft.git

ENV RAFT_PATH=/raft
ENV RMM_PATH=/rmm

RUN git clone https://github.com/NVIDIA/spark-rapids-ml.git \
    && cd spark-rapids-ml \
    && mvn clean package install

ADD scala /workspace/
ADD pom.xml /workspace/

RUN cd /workspace/ \
    && mvn clean package

# install spark-3.1.2-bin-hadoop3.2
RUN wget --quiet \
    https://archive.apache.org/dist/spark/spark-3.1.2/spark-3.1.2-bin-hadoop3.2.tgz \
    && tar -xzf spark-3.1.2-bin-hadoop3.2.tgz -C /opt/ \
    && rm spark-3.1.2-bin-hadoop3.2.tgz

# add spark env to conf
ADD spark-env.sh /opt/spark-3.1.2-bin-hadoop3.2/conf/
ADD start-spark.sh /workspace/
ADD spark-submit.sh /workspace/

WORKDIR /workspace