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
from com.nvidia.spark.examples.mortgage.consts import *
from com.nvidia.spark.examples.mortgage.etl import etl
from com.nvidia.spark.examples.utility.utils import *
from ml.dmlc.xgboost4j.scala.spark import *
from pyspark.sql import SparkSession

def main(args, xgboost_args):
    spark = (SparkSession
        .builder
        .appName(args.mainClass)
        .getOrCreate())

    train_data, eval_data, trans_data = valid_input_data(spark, args, '', schema)
    features = [x.name for x in schema if x.name != label]

    if args.mode in [ 'all', 'train' ]:
        classifier = (XGBoostClassifier(**merge_dicts(default_params, xgboost_args))
            .setLabelCol(label)
            .setFeaturesCols(features))
        if eval_data:
            classifier.setEvalSets({ 'test': eval_data })
        if not train_data:
            print('-' * 80)
            print('Usage: training data path required when mode is all or train')
            exit(1)
        model = with_benchmark('Training', lambda: classifier.fit(train_data))

        if args.modelPath:
            writer = model.write().overwrite() if args.overwrite else model
            writer.save(args.modelPath)
    else:
        model = XGBoostClassificationModel().load(args.modelPath)

    if args.mode in [ 'all', 'transform' ]:
        def transform():
            result = model.transform(trans_data).cache()
            result.foreachPartition(lambda _: None)
            return result
        if not trans_data:
            print('-' * 80)
            print('Usage: trans data path required when mode is all or transform')
            exit(1)
        result = with_benchmark('Transformation', transform)
        show_sample(args, result, label)
        with_benchmark('Evaluation', lambda: check_classification_accuracy(result, label))

    spark.stop()
