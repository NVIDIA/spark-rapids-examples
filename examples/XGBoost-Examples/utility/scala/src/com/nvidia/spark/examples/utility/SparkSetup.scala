
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

package com.nvidia.spark.examples.utility

import org.apache.spark.sql.SparkSession

object SparkSetup {
  def apply(args: Array[String], appName: String) = {
    val builder = SparkSession.builder()
    val masterBuilder = Option(System.getenv("SPARK_MASTER")).map { master =>
      builder.master(master)
    }.getOrElse(builder)

    masterBuilder.appName(appName).getOrCreate()
  }

  def apply(args: Array[String]): SparkSession = SparkSetup(args, "default")

}
