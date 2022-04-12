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

package com.nvidia.spark.examples.taxi

import com.nvidia.spark.examples.utility.{XGBoostArgs, Benchmark}
import ml.dmlc.xgboost4j.scala.spark.{XGBoostRegressionModel, XGBoostRegressor}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.sql.SparkSession

// Only 2 differences between CPU and GPU. Please refer to '=== diff ==='
object GPUMain extends Taxi {

  def main(args: Array[String]): Unit = {
    val xgboostArgs = XGBoostArgs.parse(args)
    val processor = this.getClass.getSimpleName.stripSuffix("$").substring(0, 3)
    val appInfo = Seq(appName, processor, xgboostArgs.format)

    // build spark session
    val spark = SparkSession.builder()
      .appName(appInfo.mkString("-"))
      .getOrCreate()

    val benchmark = Benchmark(appInfo(0), appInfo(1), appInfo(2))

    // build data reader
    val dataReader = spark.read

    val (pathsArray, dataReadSchema, needEtl) = getDataPaths(xgboostArgs.dataPaths, xgboostArgs.isToTrain, xgboostArgs.isToTransform)

    // 0: train 1: eval 2:transform
    var datasets = pathsArray.map { paths =>
      if (paths.nonEmpty) {
        xgboostArgs.format match {
          case "csv" => Some(dataReader.option("header", xgboostArgs.hasHeader).schema(dataReadSchema).csv(paths: _*))
          case "orc" => Some(dataReader.orc(paths: _*))
          case "parquet" => Some(dataReader.parquet(paths: _*))
          case _ => throw new IllegalArgumentException("Unsupported data file format!")
        }
      } else {
        None
      }
    }

    if (needEtl) datasets = datasets.map(_.map(preProcess(_)))

    // === diff ===
    // No need to vectorize data since GPU support multiple feature columns via API 'setFeaturesCols'

    val xgbRegressionModel = if (xgboostArgs.isToTrain) {
      // build XGBoost XGBoostRegressor
      val xgbParamFinal = xgboostArgs.xgboostParams(commParamMap +
        // Add train-eval dataset if specified
        ("eval_sets" -> datasets(1).map(ds => Map("test" -> ds)).getOrElse(Map.empty))
      )
      val xgbRegressor = new XGBoostRegressor(xgbParamFinal)
        .setLabelCol(labelColName)
        .setFeaturesCol(featureNames)

      println("\n------ Training ------")
      // Shall we not log the time if it is abnormal, which is usually caused by training failure
      val (model, _) = benchmark.time("train") {
        xgbRegressor.fit(datasets(0).get)
      }
      // Save model if modelPath exists
      xgboostArgs.modelPath.foreach(path =>
        if (xgboostArgs.isOverwrite) model.write.overwrite().save(path) else model.save(path))
      model
    } else {
      XGBoostRegressionModel.load(xgboostArgs.modelPath.get)
    }

    if (xgboostArgs.isToTransform) {
      println("\n------ Transforming ------")
      var (prediction, _) = benchmark.time("transform") {
        val ret = xgbRegressionModel.transform(datasets(2).get).cache()
        ret.foreachPartition((_: Iterator[_]) => ())
        ret
      }
      prediction = if (xgboostArgs.isShowFeatures) {
        prediction
      } else {
        prediction.select(labelColName, "prediction")
      }
      prediction.show(xgboostArgs.numRows)

      println("\n------Accuracy of Evaluation------")
      val evaluator = new RegressionEvaluator().setLabelCol(labelColName)
      evaluator.evaluate(prediction) match {
        case rmse if !rmse.isNaN => benchmark.value(rmse, "RMSE", "RMSE for")
        // Throw an exception when NaN ?
      }
    }

    spark.close()
  }
}
