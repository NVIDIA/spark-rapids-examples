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

package com.nvidia.spark.examples.utility

import scala.util.Properties

class Benchmark(
    appName: String,
    processor: String,
    dataFormat: String) {

  def time[R](phase: String, silent: (Any, Float) => Boolean = (_,_) => false)
             (block: => R): (R, Float) = {
    val t0 = System.currentTimeMillis
    val result = block // call-by-name
    val elapsedTimeSec = (System.currentTimeMillis - t0).toFloat / 1000
    logging(elapsedTimeSec, phase, "Elapsed time for", "s", silent(result, elapsedTimeSec))
    (result, elapsedTimeSec)
  }

  def value(value: Any, name: String = "value",  prefix: String="", suffix: String = "") = {
    logging(value, name, prefix, suffix, false)
  }

  private def logging(value: Any, name: String , prefix: String, suffix: String, silent: Boolean) = {
    if (!silent) {
      val logString = buildLogSimple(value, prefix, suffix, buildRuntimeInfo(name))
      println("\n--------------")
      println("==> Benchmark: " + logString)
      println("--------------\n")
    }
  }

  private def buildRuntimeInfo(name: String): String = {
    // Get runtime information from Environment
    val osType = Properties.envOrElse("RAPIDS_XGB_EXAMPLE_OS_TYPE", "Unknown")
    val cudaVersion = Properties.envOrElse("RAPIDS_XGB_EXAMPLE_CUDA_VERSION", "Unknown")
    val sparkVersion = Properties.envOrElse("RAPIDS_XGB_EXAMPLE_SPARK_VERSION", "Unknown")
    Seq(appName, processor, name, dataFormat, "stub", cudaVersion, osType, sparkVersion)
      .mkString(" ")
  }

  private def buildLogSimple(value: Any, prefix: String, suffix: String, runtimeInfo: String): String =
    prefix + " [" + runtimeInfo + "]: " + value + suffix
}

object Benchmark {
  def apply(appName: String, processor: String, dataFormat: String) =
    new Benchmark(appName, processor, dataFormat)
}
