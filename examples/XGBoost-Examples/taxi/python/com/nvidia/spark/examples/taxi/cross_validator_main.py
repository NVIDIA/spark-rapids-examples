#
# Copyright (c) 2019-2021, NVIDIA CORPORATION. All rights reserved.
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
from .consts import *
from com.nvidia.spark.examples.utility.utils import *
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql import SparkSession

from xgboost.spark import SparkXGBRegressor, SparkXGBRegressorModel


def main(args, xgboost_args):
    spark = (SparkSession
             .builder
             .appName(args.mainClass)
             .getOrCreate())

    train_data, eval_data, trans_data = valid_input_data(spark, args, raw_schema, final_schema)

    if args.mode in ['all', 'train']:
        if train_data is None:
            print('-' * 80)
            print('Usage: training data path required when mode is all or train')
            print('-' * 80)
            exit(1)

        train_data, features = transform_data(train_data, label, args.use_gpu)
        xgboost_args['features_col'] = features
        xgboost_args['label_col'] = label

        regressor = SparkXGBRegressor(**xgboost_args)

        param_grid = (ParamGridBuilder()
                      .addGrid(regressor.max_depth, [6, 8])
                      .addGrid(regressor.n_estimators, [20, 40])
                      .build())

        evaluator = (RegressionEvaluator()
                     .setLabelCol(label))

        cross_validator = (CrossValidator()
                           .setEstimator(regressor)
                           .setEvaluator(evaluator)
                           .setEstimatorParamMaps(param_grid)
                           .setNumFolds(3))

        model = with_benchmark('Training', lambda: cross_validator.fit(train_data))
        # get the best model to do transform
        model = model.bestModel
        if args.modelPath:
            writer = model.write().overwrite() if args.overwrite else model
            writer.save(args.modelPath)
    else:
        model = SparkXGBRegressorModel.load(args.modelPath)

    if args.mode in ['all', 'transform']:
        if trans_data is None:
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
        with_benchmark('Evaluation', lambda: check_regression_accuracy(result, label))

    spark.stop()
