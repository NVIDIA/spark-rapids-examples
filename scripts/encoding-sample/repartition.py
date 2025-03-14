# Copyright (c) 2024, NVIDIA CORPORATION.
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

# Note: Plase modify the data source options for your case.

import sys

from pyspark.sql import SparkSession

(SparkSession
    .builder
    .getOrCreate()
    .read
    .option('sep', '\t')
    .csv(sys.argv[1])
    .repartition(int(sys.argv[3]))
    .write
    .option('sep', '\t')
    .option('nullValue', None)
    .csv(sys.argv[2]))
