// Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// Licensed to the Apache Software Foundation (ASF) under one or more
// contributor license agreements.  See the NOTICE file distributed with
// this work for additional information regarding copyright ownership.
// The ASF licenses this file to You under the Apache License, Version 2.0
// (the "License"); you may not use this file except in compliance with
// the License.  You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._

object connect extends Serializable {

  def explain(df: DataFrame): Unit = {
    // workaround to get a plan with GpuOverrides applied by disabling adaptive execution
    df.sparkSession.conf.set("spark.sql.adaptive.enabled", "false")
    df.explain(mode = "extended")
    df.sparkSession.conf.set("spark.sql.adaptive.enabled", "true")
  }

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().getOrCreate()
    val df = spark.range(1L << 12)
      .withColumn("mod10", col("id") % lit(10))
      .groupBy("mod10").agg(count("*"))
      .orderBy("mod10")

    // Disable GPU accelerating
    println("--------------- CPU running by disabling spark.rapids.sql.enabled ---------------")
    spark.conf.set("spark.rapids.sql.enabled", "false")
    explain(df)
    df.show()

    // Enable GPU accelerating
    spark.conf.set("spark.rapids.sql.enabled", "true")
    println("--------------- GPU running by enabling spark.rapids.sql.enabled ---------------")
    explain(df)
    df.show()

    spark.stop()
  }
}
