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
import org.apache.hadoop.fs.Path
import org.apache.spark.sql.SparkSession

object ETLMain extends Mortgage {

  def main(args: Array[String]): Unit = {
    val xgbArgs = XGBoostArgs(args)
    val subTitle = getClass.getSimpleName.stripSuffix("$").substring(0, 3)
    val appInfo = Seq(appName, subTitle, xgbArgs.format)
    val benchmark = Benchmark(appInfo(0), appInfo(1), appInfo(2))
    // build spark session
    val spark = SparkSession.builder().appName(appInfo.mkString("-")).getOrCreate()

    try {
      val (perfPaths, acqPaths, outPath) = checkAndGetPaths(xgbArgs.dataPaths)
      println("\n------ Start ETL ------")
      benchmark.time("ETL") {
        // ETL the raw data
        val rawDF = xgbArgs.format match {
          case "csv" => XGBoostETL.csv(spark, perfPaths, acqPaths, xgbArgs.hasHeader)
          case "orc" => XGBoostETL.orc(spark, perfPaths, acqPaths)
          case "parquet" => XGBoostETL.parquet(spark, perfPaths, acqPaths)
          case _ => throw new IllegalArgumentException("Unsupported data file format!")
        }
        rawDF.write.mode("overwrite").parquet(new Path(outPath, "data").toString)
      }
      if (xgbArgs.saveDict) {
        XGBoostETL.saveDictTable(new Path(outPath, ".dict").toString)
      }
    } finally {
      XGBoostETL.clean()
      spark.close()
    }
  }

  private def checkAndGetPaths(paths: Seq[String]): (Seq[String], Seq[String], String) = {
    val prefixes = Array("perf::", "acq::", "out::")
    val validPaths = paths.filter(_.nonEmpty).map(_.trim)

    // get and check perf data paths
    val perfPaths = validPaths.filter(_.startsWith(prefixes.head))
    require(perfPaths.nonEmpty, s"$appName ETL requires at least one path for performance data file." +
      s" Please specify it by '-dataPath=perf::your_perf_path'")

    // get and check acq data paths
    val acqPaths = validPaths.filter(_.startsWith(prefixes(1)))
    require(acqPaths.nonEmpty, s"$appName ETL requires at least one path for acquisition data file." +
      s" Please specify it by '-dataPath=acq::your_acq_path'")

    // get and check out path
    val outPath = validPaths.filter(_.startsWith(prefixes(2)))
    require(outPath.nonEmpty, s"$appName ETL requires a path to save the ETLed data file. Please specify it" +
      " by '-dataPath=out::your_out_path', only the first path is used if multiple paths are found.")

    // check data paths not specified type
    val unknownPaths = validPaths.filterNot(p => prefixes.exists(p.contains(_)))
    require(unknownPaths.isEmpty, s"Unknown type for data path: ${unknownPaths.head}, $appName requires to specify" +
      " the type for each data path by adding the prefix 'perf::' or 'acq::' or 'out::'.")

    (perfPaths.map(_.stripPrefix(prefixes.head)),
     acqPaths.map(_.stripPrefix(prefixes(1))),
     outPath.head.stripPrefix(prefixes(2)))
  }
}
