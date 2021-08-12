/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.nvidia.spark.examples.mortgage

import com.nvidia.spark.examples.utility.{XGBoostArgs, Benchmark}
import ml.dmlc.xgboost4j.scala.spark.rapids.CrossValidator
import ml.dmlc.xgboost4j.scala.spark.{XGBoostClassificationModel, XGBoostClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.sql.SparkSession

// Only 2 differences between CPU and GPU. Please refer to '=== diff ==='
object GPUCrossValidationMain extends Mortgage {

  def main(args: Array[String]): Unit = {
    val appArgs = XGBoostArgs(args)
    val processor = this.getClass.getSimpleName.stripSuffix("$").substring(0, 3)
    val appInfo = Seq(appName, processor, appArgs.format)
    val benchmark = Benchmark(appInfo(0), appInfo(1), appInfo(2))
    // build spark session
    val spark = SparkSession.builder().appName(appInfo.mkString("-")).getOrCreate()
    // build data reader
    val dataReader = spark.read

    try {
      // loaded XGBoost ETLed data
      val pathsArray = appArgs.getDataPaths
      // 0: train 1: eval 2:transform
      val datasets = pathsArray.map { paths =>
        if (paths.nonEmpty) {
          appArgs.format match {
            case "csv" => Some(dataReader.option("header", appArgs.hasHeader).schema(schema).csv(paths: _*))
            case "orc" => Some(dataReader.orc(paths: _*))
            case "parquet" => Some(dataReader.parquet(paths: _*))
            case _ => throw new IllegalArgumentException("Unsupported data file format!")
          }
        } else {
          None
        }
      }

      // === diff ===
      // No need to vectorize data since GPU support multiple feature columns via API 'setFeaturesCols'

      val xgbClassificationModel = if (appArgs.isToTrain) {
        // build XGBoost classifier
        val xgbParamFinal = appArgs.xgboostParams(commParamMap)
        val xgbClassifier = new XGBoostClassifier(xgbParamFinal)
          .setLabelCol(labelColName)
          // === diff ===
          .setFeaturesCols(featureNames)

        // Tune model using cross validation
        val paramGrid = new ParamGridBuilder()
          .addGrid(xgbClassifier.maxDepth, Array(3, 10))
          .addGrid(xgbClassifier.eta, Array(0.2, 0.6))
          .build()
        val evaluator = new MulticlassClassificationEvaluator().setLabelCol(labelColName)

        val cv = new CrossValidator()
          .setEstimator(xgbClassifier)
          .setEvaluator(evaluator)
          .setEstimatorParamMaps(paramGrid)
          .setNumFolds(appArgs.numFold)

        // Start training
        println("\n------ CrossValidation ------")
        // Shall we not log the time if it is abnormal, which is usually caused by training failure
        val (model, _) = benchmark.time("CrossValidation") {
          cv.fit(datasets(0).get).bestModel.asInstanceOf[XGBoostClassificationModel]
        }
        // Save model if modelPath exists
        appArgs.modelPath.foreach(path =>
          if (appArgs.isOverwrite) model.write.overwrite().save(path) else model.save(path))
        model
      } else {
        XGBoostClassificationModel.load(appArgs.modelPath.get)
      }

      if (appArgs.isToTransform) {
        println("\n------ Transforming ------")
        var (results, _) = benchmark.time("transform") {
          val ret = xgbClassificationModel.transform(datasets(2).get).cache()
          // Trigger the transformation
          ret.foreachPartition((_: Iterator[_]) => ())
          ret
        }
        results = if (appArgs.isShowFeatures) {
          results
        } else {
          results.select(labelColName, "rawPrediction", "probability", "prediction")
        }
        results.show(appArgs.numRows)

        println("\n------Accuracy of Evaluation------")
        val evaluator = new MulticlassClassificationEvaluator().setLabelCol(labelColName)
        evaluator.evaluate(results) match {
          case accuracy if !accuracy.isNaN =>
            benchmark.value(accuracy, "Accuracy", "Accuracy for")
          // Throw an exception when NaN ?
        }
      }
    } finally {
      spark.close()
    }
  }
}
