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
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType, DoubleType
import time


def get_dict_df(train_df, target_col, label_col):
    '''
    get one dict dataframe for one column
    '''
    col_target_df = train_df.groupBy(target_col).agg(F.mean(label_col))
    return col_target_df

def encode_df(original_df, dict_df, col_name):
    dict_df_rename = dict_df.withColumnRenamed('_c0', 'hash').withColumnRenamed('_c1', col_name+'_mean')
    df_mean = (original_df.join(dict_df_rename, original_df[col_name] == dict_df_rename['hash'], how='left').drop('hash').drop(col_name)
        .na.fill(-1, [col_name + '_mean']))
    return df_mean


def main(args):
    spark = (SparkSession
        .builder
        .appName(args.mainClass)
        .getOrCreate())
    if args.mode == 'train':
        for col_name, model_path in zip(args.columns, args.modelPaths):
            df = load_data(spark, args.inputPaths, args, customize_reader).cache()
            dict_df = get_dict_df(df, col_name, args.labelColumn)
            dict_df.repartition(1).write.csv(model_path)

    if args.mode == 'transform':
        dict_dfs = [
            load_dict_df(spark, path).withColumn('_c1', F.col('_c1').cast(DoubleType())).cache()
            for path in args.modelPaths
        ]
        for input_path, output_path in zip(args.inputPaths, args.outputPaths):
            df = load_data(spark, input_path, args, customize_reader)
            for col_name, dict_df in zip(args.columns, dict_dfs):
                df = encode_df(df, dict_df, col_name)
            save_data(df, output_path, args, customize_writer)