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
from pyspark.sql.types import *

from com.nvidia.spark.examples.utility.utils import *
from pyspark.sql import SparkSession

from xgboost.spark import SparkXGBClassifier, SparkXGBClassifierModel

label = 'label'
feature_names = ['feature_' + str(i) for i in range(0, 126)]
schema = StructType([StructField(x, FloatType()) for x in [label] + feature_names])


def main(args, xgboost_args):
    spark = (SparkSession
             .builder
             .appName(args.mainClass)
             .getOrCreate())

    train_data, eval_data, trans_data = valid_input_data(spark, args, '', schema)

    if args.mode in ['all', 'train']:
        if train_data is not None:
            print('-' * 80)
            print('Usage: train data path required when mode is all or train')
            print('-' * 80)
            exit(1)

        train_data, features = transform_data(train_data, label, args.use_gpu)
        xgboost_args['features_col'] = features
        xgboost_args['label_col'] = label
        classifier = SparkXGBClassifier(**xgboost_args)

        if eval_data:
            # TODO
            pass

        model = with_benchmark('Training', lambda: classifier.fit(train_data))

        if args.modelPath:
            writer = model.write().overwrite() if args.overwrite else model
            writer.save(args.modelPath)
    else:
        model = SparkXGBClassifierModel.load(args.modelPath)

    if args.mode in ['all', 'transform']:
        if trans_data is not None:
            print('-' * 80)
            print('Usage: trans data path required when mode is all or transform')
            print('-' * 80)
            exit(1)

        trans_data, _ = transform_data(trans_data, label, args.use_gpu)

        def transform():
            result = model.transform(trans_data).cache()
            result.foreachPartition(lambda _: None)
            return result

        result = with_benchmark('Transformation', transform)
        show_sample(args, result, label)
        with_benchmark('Evaluation', lambda: check_classification_accuracy(result, label))

    spark.stop()
