!/bin/bash

# install dependencies for petastorm and nvtabular data loaders
/databricks/python/bin/pip install cudf-cu11 dask-cudf-cu11 --extra-index-url=https://pypi.nvidia.com
/databricks/python/bin/pip install \
    merlin-dataloader \
    nvtabular \
    "numpy<1.24" \
    "protobuf<3.20" \
    "pyarrow==10.0.1" \
    s3fs \
    tritonclient

# https://stackoverflow.com/questions/71759248/importerror-cannot-import-name-builder-from-google-protobuf-internal
wget https://raw.githubusercontent.com/protocolbuffers/protobuf/main/python/google/protobuf/internal/builder.py -O /databricks/python/lib/python3.10/site-packages/google/protobuf/internal/builder.py
