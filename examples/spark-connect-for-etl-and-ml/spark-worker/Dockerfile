FROM apache/spark:4.0.0

USER root
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --extra-index-url https://pypi.anaconda.org/rapidsai-wheels-nightly/simple --pre -r /tmp/requirements.txt

# TODO hack to avoid configuring cupy compiler path
RUN mkdir -p /home/spark
RUN chown -R spark:spark /home/spark
RUN usermod -d /home/spark spark

USER spark
COPY spark-env.sh /opt/spark/conf/spark-env.sh
