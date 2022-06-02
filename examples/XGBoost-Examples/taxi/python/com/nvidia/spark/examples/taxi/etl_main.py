#
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from com.nvidia.spark.examples.taxi.consts import *
from com.nvidia.spark.examples.taxi.pre_process import pre_process
from com.nvidia.spark.examples.utility.utils import *
from ml.dmlc.xgboost4j.scala.spark import *
from pyspark.sql import SparkSession

def main(args, xgboost_args):
    spark = (SparkSession
        .builder
        .appName(args.mainClass)
        .getOrCreate())
    raw_data_path = extract_paths(args.dataPaths, 'raw::')
    output_path = extract_paths(args.dataPaths, 'out::')[0]
    if not raw_data_path:
        print('-' * 80)
        print('Usage: raw data path required when ETL')
        exit(1)
    if not output_path:
        print('-' * 80)
        print('Usage: output data path required when ETL')
        exit(1)
    raw_data = prepare_data(spark, args, raw_schema, raw_data_path)
    etled_train, etled_eval, etled_trans = pre_process(raw_data).randomSplit(list(map(float, args.splitRatios)))
    etled_train.write.mode("overwrite").parquet(output_path+'/train')
    etled_eval.write.mode("overwrite").parquet(output_path+'/eval')
    etled_trans.write.mode("overwrite").parquet(output_path+'/trans')
