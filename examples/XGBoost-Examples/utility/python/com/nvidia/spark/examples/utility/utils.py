#
# Copyright (c) 2019-2022, NVIDIA CORPORATION. All rights reserved.
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
import typing

from pyspark.ml.evaluation import *
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import DataFrame
from pyspark.sql.functions import col
from pyspark.sql.types import FloatType
from com.nvidia.spark.examples.taxi.pre_process import pre_process
from time import time


def merge_dicts(dict_x, dict_y):
    result = dict_x.copy()
    result.update(dict_y)
    return result


def show_sample(args, data_frame, label):
    data_frame = data_frame if args.showFeatures else data_frame.select(label, 'prediction')
    data_frame.show(args.numRows)


def vectorize_data_frame(data_frame, label):
    features = [x.name for x in data_frame.schema if x.name != label]
    to_floats = [col(x.name).cast(FloatType()) for x in data_frame.schema]
    return (VectorAssembler()
            .setInputCols(features)
            .setOutputCol('features')
            .transform(data_frame.select(to_floats))
            .select(col('features'), col(label)))


def vectorize_data_frames(data_frames, label):
    return [vectorize_data_frame(x, label) for x in data_frames]


def with_benchmark(phrase, action):
    start = time()
    result = action()
    end = time()
    print('-' * 100)
    print('{} takes {} seconds'.format(phrase, round(end - start, 2)))
    return result


def check_classification_accuracy(data_frame, label):
    accuracy = (MulticlassClassificationEvaluator()
                .setLabelCol(label)
                .evaluate(data_frame))
    print('-' * 100)
    print('Accuracy is ' + str(accuracy))


def check_regression_accuracy(data_frame, label):
    accuracy = (RegressionEvaluator()
                .setLabelCol(label)
                .evaluate(data_frame))
    print('-' * 100)
    print('RMSE is ' + str(accuracy))


def prepare_data(spark, args, schema, dataPath):
    reader = (spark
              .read
              .format(args.format))
    if args.format == 'csv':
        reader.schema(schema).option('header', args.hasHeader)
    return reader.load(dataPath)


def extract_paths(paths, prefix):
    results = [path[len(prefix):] for path in paths if path.startswith(prefix)]
    return results


def transform_data(
        df: DataFrame,
        label: str,
        use_gpu: typing.Optional[bool],
) -> (DataFrame, typing.Union[str, typing.List[str]]):
    if use_gpu:
        features = [x.name for x in df.schema if x.name != label]
    else:
        df = vectorize_data_frame(df, label)
        features = 'features'
    return df, features


def valid_input_data(spark, args, raw_schema, final_schema):
    e2e = False
    for path in args.dataPaths:
        if 'raw' in path:
            e2e = True
            break
    raw_train_path = ''
    raw_eval_path = ''
    raw_trans_path = ''
    eval_path = ''

    if e2e:
        raw_train_path = extract_paths(args.dataPaths, 'rawTrain::')
        raw_eval_path = extract_paths(args.dataPaths, 'rawEval::')
        raw_trans_path = extract_paths(args.dataPaths, 'rawTrans::')

    train_data = ''
    eval_data = ''
    trans_data = ''

    # if this is an e2e run
    if raw_train_path or raw_eval_path or raw_trans_path:
        raw_train_data = prepare_data(spark, args, raw_schema, raw_train_path)
        raw_eval_data = ''
        raw_trans_data = ''
        if raw_eval_path:
            raw_eval_data = prepare_data(spark, args, raw_schema, raw_eval_path)
        if raw_trans_path:
            raw_trans_data = prepare_data(spark, args, raw_schema, raw_trans_path)

        train_data = pre_process(raw_train_data)
        if raw_eval_data:
            eval_data = pre_process(raw_eval_data)
        if raw_trans_data:
            trans_data = pre_process(raw_trans_data)

    # if this is just a train/transform
    else:
        train_path = extract_paths(args.dataPaths, 'train::')
        eval_path = extract_paths(args.dataPaths, 'eval::')
        trans_path = extract_paths(args.dataPaths, 'trans::')
        if train_path:
            train_data = prepare_data(spark, args, final_schema, train_path)
        if eval_path:
            eval_data = prepare_data(spark, args, final_schema, eval_path)
        if trans_path:
            trans_data = prepare_data(spark, args, final_schema, trans_path)
    return (train_data, eval_data, trans_data)
