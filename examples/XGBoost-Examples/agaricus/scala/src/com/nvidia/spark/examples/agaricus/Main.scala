/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION. All rights reserved.
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

package com.nvidia.spark.examples.agaricus

import com.nvidia.spark.examples.utility.{Benchmark, SparkSetup, XGBoostArgs}
import ml.dmlc.xgboost4j.scala.spark.{XGBoostClassificationModel, XGBoostClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.types.{FloatType, StructField, StructType}

object Main {
  def main(args: Array[String]): Unit = {

    val labelName = "label"

    def featureNames(length: Int): List[String] =
      0.until(length).map(i => s"feature_$i").toList.+:(labelName)

    def schema(length: Int): StructType =
      StructType(featureNames(length).map(n => StructField(n, FloatType)))

    val dataSchema = schema(126)
    val xgboostArgs = XGBoostArgs.parse(args)
    val processor = this.getClass.getSimpleName.stripSuffix("$").substring(0, 3)
    val appInfo = Seq("Agaricus", processor, xgboostArgs.format)

    // build spark session
    val spark = SparkSetup(args, appInfo.mkString("-"))
    val benchmark = Benchmark(appInfo(0), appInfo(1), appInfo(2))

    // build data reader
    val dataReader = spark.read

    // load data
    val pathsArray = xgboostArgs.getDataPaths
    // train, eval, transform
    var datasets = pathsArray.map { paths =>
      if (paths.nonEmpty) {
        xgboostArgs.format match {
          case "csv" => Some(dataReader.option("header", xgboostArgs.hasHeader).schema(dataSchema).csv(paths: _*))
          case "orc" => Some(dataReader.orc(paths: _*))
          case "parquet" => Some(dataReader.parquet(paths: _*))
          case _ => throw new IllegalArgumentException("Unsupported data file format!")
        }
      } else None
    }

    val featureCols = dataSchema.filter(_.name != labelName).map(_.name).toArray

    val xgbClassificationModel = if (xgboostArgs.isToTrain) {
      // build XGBoost classifier
      val paramMap = xgboostArgs.xgboostParams(Map(
        "objective" -> "binary:logistic",
      ))
      val xgbClassifier = new XGBoostClassifier(paramMap)
        .setLabelCol(labelName)
        // === diff ===
        .setFeaturesCol(featureCols)

      datasets(1).foreach(_ => xgbClassifier.setEvalDataset(_))

      println("\n------ Training ------")
      val (model, _) = benchmark.time("train") {
        xgbClassifier.fit(datasets(0).get)
      }
      // Save model if modelPath exists
      xgboostArgs.modelPath.foreach(path =>
        if (xgboostArgs.isOverwrite) model.write.overwrite().save(path) else model.save(path))
      model
    } else {
      XGBoostClassificationModel.load(xgboostArgs.modelPath.get)
    }

    if (xgboostArgs.isToTransform) {
      // start transform
      println("\n------ Transforming ------")
      var (results, _) = benchmark.time("transform") {
        val ret = xgbClassificationModel.transform(datasets(2).get).cache()
        ret.foreachPartition((_: Iterator[_]) => ())
        ret
      }
      results = if (xgboostArgs.isShowFeatures) {
        results
      } else {
        results.select(labelName, "rawPrediction", "probability", "prediction")
      }
      results.show(xgboostArgs.numRows)

      println("\n------Accuracy of Evaluation------")
      val evaluator = new MulticlassClassificationEvaluator().setLabelCol(labelName)
      evaluator.evaluate(results) match {
        case accuracy if !accuracy.isNaN =>
          benchmark.value(accuracy, "Accuracy", "Accuracy for")
        // Throw an exception when NaN ?
      }
    }

    spark.close()
  }
}
