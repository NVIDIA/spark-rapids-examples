FROM apache/spark:4.0.0

USER root
COPY requirements.txt /opt/spark/work-dir/requirements.txt
RUN pip3 install --extra-index-url https://pypi.anaconda.org/rapidsai-wheels-nightly/simple --pre -r /opt/spark/work-dir/requirements.txt

USER spark
RUN wget -q -P /tmp https://edge.urm.nvidia.com/artifactory/sw-spark-maven/com/nvidia/rapids-4-spark_2.13/25.08.0/rapids-4-spark_2.13-25.08.0-cuda12.jar
COPY spark-defaults.conf /opt/spark/conf/spark-defaults.conf

