/*
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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
import org.apache.spark.sql.SparkSession

object ETLMain extends Taxi {

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

    val (rawPaths, outPath) = checkAndGetPaths(xgboostArgs.dataPaths)
    val df = xgboostArgs.format match {
      case "csv" => dataReader.option("header", xgboostArgs.hasHeader).schema(rawSchema).csv(rawPaths: _*)
      case "parquet" => dataReader.parquet(rawPaths: _*)
      case "orc" => dataReader.orc(rawPaths: _*)
      case _ => throw new IllegalArgumentException("Unsupported data file format!")
    }

    val (trainRatio, evalRatio, trainEvalRatio) = xgboostArgs.dataRatios
    val dataset = preProcess(df, Array(trainRatio, trainEvalRatio, evalRatio))

    benchmark.time("ETL") {
      for ((name, index) <- Seq("train", "eval", "trans").zipWithIndex) {
        dataset(index).write.mode("overwrite").parquet(outPath + "/parquet/" + name)
        dataset(index).write.mode("overwrite").csv(outPath + "/csv/" + name)
      }
    }

    spark.close()
  }

  private def checkAndGetPaths(paths: Seq[String]): (Seq[String], String) = {
    val prefixes = Array("raw::", "out::")
    val validPaths = paths.filter(_.nonEmpty).map(_.trim)

    // get and check train data paths
    val rawPaths = validPaths.filter(_.startsWith(prefixes.head))
    require(rawPaths.nonEmpty, s"$appName ETL requires at least one path for taxi data file." +
      s" Please specify it by '-dataPath=raw::your_taxi_data_path'")

    // get and check out path
    val outPath = validPaths.filter(_.startsWith(prefixes(1)))
    require(outPath.nonEmpty, s"$appName ETL requires a path to save the ETLed data file. Please specify it" +
      " by '-dataPath=out::your_out_path', only the first path is used if multiple paths are found.")

    // check data paths not specified type
    val unknownPaths = validPaths.filterNot(p => prefixes.exists(p.contains(_)))
    require(unknownPaths.isEmpty, s"Unknown type for data path: ${unknownPaths.head}, $appName requires to specify" +
      " the type for each data path by adding the prefix 'raw::' or 'out::'")

    (rawPaths.map(_.stripPrefix(prefixes.head)),
      outPath.head.stripPrefix(prefixes(1)))
  }
}
