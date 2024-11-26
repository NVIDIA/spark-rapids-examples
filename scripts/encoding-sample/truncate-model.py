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

import sys

from pyspark.sql import SparkSession
from pyspark.sql.functions import *

(SparkSession
    .builder
    .getOrCreate()
    .read
    .csv(sys.argv[1])
    .withColumn('_c1', format_string('%.6f', col('_c1').cast('float')))
    .withColumn('_c1', when(col('_c1') == '0.000000', lit('0.0')).otherwise(col('_c1')))
    .withColumn('_c1', when(col('_c1') == '1.000000', lit('1.0')).otherwise(col('_c1')))
    .repartition(1)
    .write
    .option('nullValue', None)
    .csv(sys.argv[2]))
