FROM jupyter/minimal-notebook:latest

COPY notebooks /home/jovyan/work
RUN pip install --extra-index-url https://pypi.anaconda.org/rapidsai-wheels-nightly/simple --pre \
  -r /home/jovyan/work/requirements.txt
