FROM apache/spark:4.0.0

COPY spark-env.sh /opt/spark/conf/spark-env.sh
COPY requirements.txt /opt/spark/work-dir/requirements.txt

USER root
RUN pip3 install --extra-index-url https://pypi.anaconda.org/rapidsai-wheels-nightly/simple --pre -r /opt/spark/work-dir/requirements.txt

USER spark