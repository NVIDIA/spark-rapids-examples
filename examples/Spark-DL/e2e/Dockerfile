FROM nvidia/cuda:11.2.1-cudnn8-devel-ubuntu20.04


ENV DEBIAN_FRONTEND=noninteractive
# Disable NVIDIA repos to prevent accidental upgrades.
RUN cd /etc/apt/sources.list.d && \
    mv cuda.list cuda.list.disabled && \
    mv nvidia-ml.list nvidia-ml.list.disabled

# See https://github.com/databricks/containers/blob/master/ubuntu/minimal/Dockerfile
RUN apt-get update && \
    apt-get install --yes --no-install-recommends \
      openjdk-8-jdk \
      openjdk-8-jre \
      lsb-release \
      iproute2 \
      bash \
      sudo \
      coreutils \
      procps \
      wget && \
    /var/lib/dpkg/info/ca-certificates-java.postinst configure && \
    rm -rf /var/lib/apt/lists/*


ENV PATH /databricks/conda/bin:$PATH

RUN wget -q https://repo.continuum.io/miniconda/Miniconda3-py38_4.9.2-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /databricks/conda && \
    rm miniconda.sh && \
    # Source conda.sh for all login and interactive shells.
    ln -s /databricks/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /etc/profile.d/conda.sh" >> ~/.bashrc && \
    # Set always_yes for non-interactive shells.
    conda config --system --set always_yes True && \
    conda clean --all

RUN pip uninstall tensorflow -y; pip install tf-nightly==2.7.0.dev20210722
RUN pip uninstall keras-nightly -y; pip install keras-nightly==2.7.0.dev2021072200

RUN mkdir -p /usr/local/nvidia/lib64 && \
    ln -s /usr/local/cuda/lib64/libcusolver.so /usr/local/nvidia/lib64/libcusolver.so.10

# install openjdk8, cmake, openmpi openmpi-mpicc
RUN conda install cmake=3.19.6 openmpi openmpi-mpicc -y 

RUN HOROVOD_WITH_MPI=1 HOROVOD_WITH_TENSORFLOW=1 HOROVOD_GPU_OPERATIONS=NCCL \
    pip install horovod[spark] --no-cache-dir

RUN conda install -c nvidia -c rapidsai -c numba -c conda-forge nvtabular=0.6.1 python=3.8 cudatoolkit=11.2

RUN pip install pynvml

RUN conda install ipython==7.19.0 matplotlib==3.4.2 jinja2==2.11.3
RUN pip uninstall pandas -y; pip install pandas==1.1.5
RUN apt-get update && apt-get install wget openssh-client openssh-server \
    -y --allow-downgrades --allow-change-held-packages --no-install-recommends

RUN pip install jupyter
# Download Spark 
RUN wget -q https://urm.nvidia.com/artifactory/sw-spark-maven/org/apache/spark/3.1.2/spark-3.1.2-bin-hadoop3.2.tgz && \
    tar -xzf spark-3.1.2-bin-hadoop3.2.tgz && \
    rm spark-3.1.2-bin-hadoop3.2.tgz

ENV JAVA_HOME /usr/lib/jvm/java-1.8.0-openjdk-amd64
ENV PATH $PATH:/usr/lib/jvm/java-1.8.0-openjdk-amd64/jre/bin:/usr/lib/jvm/java-1.8.0-openjdk-amd64/bin

# add spark env to conf
ADD spark-env.sh /spark-3.1.2-bin-hadoop3.2/conf/
RUN mkdir /root/.jupyter
ADD jupyter_notebook_config.py /root/.jupyter/

RUN wget -q https://urm.nvidia.com/artifactory/sw-spark-maven/ai/rapids/cudf/22.02.0-SNAPSHOT/cudf-22.02.0-20211222.092217-36-cuda11.jar -P /spark-3.1.2-bin-hadoop3.2/jars/
RUN wget -q https://urm.nvidia.com/artifactory/sw-spark-maven/com/nvidia/rapids-4-spark_2.12/22.02.0-SNAPSHOT/rapids-4-spark_2.12-22.02.0-20211221.164655-25.jar -P /spark-3.1.2-bin-hadoop3.2/jars/

ADD launch.sh /launch.sh
ADD start-spark.sh /start-spark.sh