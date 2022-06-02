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
#
from com.nvidia.spark.encoding.criteo.common import *
from com.nvidia.spark.encoding.utility.utils import *
from pyspark.ml.feature import StringIndexer, StringIndexerModel
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

def index(df, column):
    column_index = column + '_index'
    return (StringIndexer(inputCol=column, outputCol=column_index)
        .setHandleInvalid('keep')
        .fit(df))

def expand(indexer, df, column):
    column_index = column + '_index'
    df = (indexer
        .transform(df)
        .withColumn(column_index, col(column_index).cast('int')))
    for i in range(0, len(indexer.labels)):
        df = df.withColumn(column + '_' + str(i), (col(column_index) == i).cast('int'))
    return df.drop(column, column_index)

def main(args):
    spark = (SparkSession
        .builder
        .appName(args.mainClass)
        .getOrCreate())

    if args.mode == 'train':
        df = load_data(spark, args.inputPaths, args, customize_reader).cache()
        for column, path in zip(args.columns, args.modelPaths):
            indexer = index(df, column)
            save_model(indexer, path, args)

    if args.mode == 'transform':
        indexers = list(zip(args.columns, load_models(StringIndexerModel, args.modelPaths)))
        for input_path, output_path in zip(args.inputPaths, args.outputPaths):
            df = load_data(spark, input_path, args, customize_reader)
            for column, indexer in indexers:
                df = expand(indexer, df, column)
            args.numRows and df.show(args.numRows)
            save_data(df, output_path, args, customize_writer)

    spark.stop()
