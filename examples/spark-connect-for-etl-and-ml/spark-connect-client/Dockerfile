FROM jupyter/minimal-notebook:latest

RUN pip install --extra-index-url https://pypi.anaconda.org/rapidsai-wheels-nightly/simple --pre \
  pyspark-client==4.0.0

COPY notebooks /home/jovyan/work