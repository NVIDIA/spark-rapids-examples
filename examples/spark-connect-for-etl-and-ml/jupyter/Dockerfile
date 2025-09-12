FROM jupyter/minimal-notebook:latest

COPY requirements.txt /tmp/requirements.txt
RUN pip install --extra-index-url https://pypi.anaconda.org/rapidsai-wheels-nightly/simple --pre -r /tmp/requirements.txt
