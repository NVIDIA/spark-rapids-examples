#!/bin/bash
# Copyright (c) 2024-2025, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# clear
rm -f encoding.zip main.py
rm -f raw-*.csv
rm -rf model target-* onehot-* final-*

# prepare data
head -n 500 ../../datasets/clicklog.csv > raw-1.csv
head -n 750 ../../datasets/clicklog.csv | tail -n 250 > raw-2.csv
tail -n 250 ../../datasets/clicklog.csv > raw-3.csv

# assemble python libs
pushd ../encoding/python/
zip -r ../../encoding-sample/encoding.zip ai/
cp main.py ../../encoding-sample/
popd

# train target models/dicts
spark-submit --py-files encoding.zip main.py \
    --mainClass=com.nvidia.spark.encoding.criteo.target_cpu_main --mode=train \
    --format=csv --inputPaths=raw-1.csv,raw-2.csv \
    --labelColumn=_c0 --columns=_c34,_c35 --modelPaths=model/c34.dict,model/c35.dict
spark-submit truncate-model.py model/c34.dict model/c34_truncated.dict
spark-submit truncate-model.py model/c35.dict model/c35_truncated.dict

# train onehot models/indexers
spark-submit --py-files encoding.zip main.py \
    --mainClass=com.nvidia.spark.encoding.criteo.one_hot_cpu_main --mode=train \
    --format=csv --inputPaths=raw-1.csv,raw-2.csv \
    --columns=_c19,_c26 --modelPaths=model/_c19,model/_c26

# target encoding
spark-submit --py-files encoding.zip main.py \
    --mainClass=com.nvidia.spark.encoding.criteo.target_cpu_main --mode=transform \
    --columns=_c34,_c35 --modelPaths=model/c34_truncated.dict,model/c35_truncated.dict \
    --format=csv --inputPaths=raw-1.csv,raw-2.csv,raw-3.csv --outputPaths=target-1,target-2,target-3

# onehot encoding
# NOTE: If the column index changed after target encoding, you should change the metadata of all
#       models accordingly. E.g., change "outputCol":"_c26_index","inputCol":"_c26" to
#       "outputCol":"_c25_index","inputCol":"_c25" for file model/_c26/metadata/part-00000.
#       This is verified on Spark 2.x.
spark-submit --py-files encoding.zip main.py \
    --mainClass=com.nvidia.spark.encoding.criteo.one_hot_cpu_main --mode=transform \
    --columns=_c19,_c26 --modelPaths=model/_c19,model/_c26 \
    --format=csv --inputPaths=target-1,target-2,target-3 --outputPaths=onehot-1,onehot-2,onehot-3

# NOTE: As an example, not all categorical columns are encoded here.
#       But please encode all categorical columns in production environment.

# repartition <input path> <output path> <number of partitions>
spark-submit repartition.py onehot-1 final-1 5
spark-submit repartition.py onehot-2 final-2 5
spark-submit repartition.py onehot-3 final-3 5

# known issues:
#   - Issue: "org.apache.spark.shuffle.FetchFailedException: Too large frame: ...":
#     Solution: Add "--conf spark.maxRemoteBlockSizeFetchToMem=1G"
