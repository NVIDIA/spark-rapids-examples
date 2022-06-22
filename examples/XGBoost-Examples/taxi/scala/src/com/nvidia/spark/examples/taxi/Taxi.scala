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

package com.nvidia.spark.examples.taxi

import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.DataTypes.{DoubleType, IntegerType, StringType}
import org.apache.spark.sql.types.{FloatType, StructField, StructType}

private[taxi] trait Taxi {
  val appName = "Taxi"
  lazy val labelColName = "fare_amount"
  lazy val featureNames = etledSchema.filter(_.name != labelColName).map(_.name).toArray

  lazy val commParamMap = Map(
    "learning_rate" -> 0.05,
    "max_depth" -> 8,
    "subsample" -> 0.8,
    "gamma" -> 1
  )

  val rawSchema = StructType(Seq(
    StructField("vendor_id", StringType),
    StructField("pickup_datetime", StringType),
    StructField("dropoff_datetime", StringType),
    StructField("passenger_count", IntegerType),
    StructField("trip_distance", DoubleType),
    StructField("pickup_longitude", DoubleType),
    StructField("pickup_latitude", DoubleType),
    StructField("rate_code", StringType),
    StructField("store_and_fwd_flag", StringType),
    StructField("dropoff_longitude", DoubleType),
    StructField("dropoff_latitude", DoubleType),
    StructField("payment_type", StringType),
    StructField(labelColName, DoubleType),
    StructField("surcharge", DoubleType),
    StructField("mta_tax", DoubleType),
    StructField("tip_amount", DoubleType),
    StructField("tolls_amount", DoubleType),
    StructField("total_amount", DoubleType)
  ))

  private val etledSchema =
    StructType(Array(
      StructField("vendor_id", FloatType),
      StructField("passenger_count", FloatType),
      StructField("trip_distance", FloatType),
      StructField("pickup_longitude", FloatType),
      StructField("pickup_latitude", FloatType),
      StructField("rate_code", FloatType),
      StructField("store_and_fwd", FloatType),
      StructField("dropoff_longitude", FloatType),
      StructField("dropoff_latitude", FloatType),
      StructField(labelColName, FloatType),
      StructField("hour", FloatType),
      StructField("year", IntegerType),
      StructField("month", IntegerType),
      StructField("day", FloatType),
      StructField("day_of_week", FloatType),
      StructField("is_weekend", FloatType)
    ))

  def preProcess(dataFrame: DataFrame): DataFrame = {
    val processes = Seq[DataFrame => DataFrame](
      dropUseless,
      encodeCategories,
      fillNa,
      removeInvalid,
      convertDatetime,
      addHDistance
    )

    processes
      .foldLeft(dataFrame) { case (df, process) => process(df) }
  }

  def preProcess(dataFrame: DataFrame, splits: Array[Int]): Array[DataFrame] = {
    val processes = Seq[DataFrame => DataFrame](
      dropUseless,
      encodeCategories,
      fillNa,
      removeInvalid,
      convertDatetime,
      addHDistance
    )

    processes
      .foldLeft(dataFrame) { case (df, process) => process(df) }
      .cache()
      .randomSplit(splits.map(_.toDouble))
  }

  def dropUseless(dataFrame: DataFrame): DataFrame = {
    dataFrame.drop(
      "dropoff_datetime",
      "payment_type",
      "surcharge",
      "mta_tax",
      "tip_amount",
      "tolls_amount",
      "total_amount")
  }

  def encodeCategories(dataFrame: DataFrame): DataFrame = {
    val categories = Seq("vendor_id", "rate_code", "store_and_fwd_flag")

    (categories.foldLeft(dataFrame) {
      case (df, category) => df.withColumn(category, hash(col(category)))
    }).withColumnRenamed("store_and_fwd_flag", "store_and_fwd")
  }

  def fillNa(dataFrame: DataFrame): DataFrame = {
    dataFrame.na.fill(-1)
  }

  def removeInvalid(dataFrame: DataFrame): DataFrame = {
    val conditions = Seq(
      Seq("fare_amount", 0, 500),
      Seq("passenger_count", 0, 6),
      Seq("pickup_longitude", -75, -73),
      Seq("dropoff_longitude", -75, -73),
      Seq("pickup_latitude", 40, 42),
      Seq("dropoff_latitude", 40, 42))

    conditions
      .map { case Seq(column, min, max) => "%s > %d and %s < %d".format(column, min, column, max) }
      .foldLeft(dataFrame) {
        _.filter(_)
      }
  }

  def convertDatetime(dataFrame: DataFrame): DataFrame = {
    val datetime = col("pickup_datetime")
    dataFrame
      .withColumn("pickup_datetime", to_timestamp(datetime))
      .withColumn("year", year(datetime))
      .withColumn("month", month(datetime))
      .withColumn("day", dayofmonth(datetime))
      .withColumn("day_of_week", dayofweek(datetime))
      .withColumn(
        "is_weekend",
        col("day_of_week").isin(1, 7).cast(IntegerType)) // 1: Sunday, 7: Saturday
      .withColumn("hour", hour(datetime))
      .drop(datetime.toString)
  }

  def addHDistance(dataFrame: DataFrame): DataFrame = {
    val P = math.Pi / 180
    val lat1 = col("pickup_latitude")
    val lon1 = col("pickup_longitude")
    val lat2 = col("dropoff_latitude")
    val lon2 = col("dropoff_longitude")
    val internalValue = (lit(0.5)
      - cos((lat2 - lat1) * P) / 2
      + cos(lat1 * P) * cos(lat2 * P) * (lit(1) - cos((lon2 - lon1) * P)) / 2)
    val hDistance = lit(12734) * asin(sqrt(internalValue))
    dataFrame.withColumn("h_distance", hDistance)
  }

  /**
   * getDataPaths check and get train/eval/transform paths
   *
   * @return Array(train_paths, eval_paths, transform_paths)
   */
  def getDataPaths(dataPaths: Seq[String], isToTrain: Boolean, isToTransform: Boolean):
  (Array[Seq[String]], StructType, Boolean) = {
    val paths = dataPaths
    val etledPrefixes = Array("train::", "eval::", "trans::")
    val rawPrefixes = Array("rawTrain::", "rawEval::", "rawTrans::")
    val validPaths = paths.filter(_.nonEmpty).map(_.trim)

    val p1 = validPaths.filter(p => etledPrefixes.exists(p.startsWith(_)))
    val p2 = validPaths.filter(p => rawPrefixes.exists(p.startsWith(_)))

    require(p1.isEmpty || p2.isEmpty, s"requires directly train by '-dataPath=${etledPrefixes(0)}train_data_path" +
      s" -dataPath=${etledPrefixes(1)}eval_data_path -dataPath=${etledPrefixes(2)}transform_data_path' Or " +
      s"E2E train by '-dataPath=${rawPrefixes(0)}train_data_path -dataPath=${rawPrefixes(1)}eval_data_path" +
      s" -dataPath=${rawPrefixes(2)}transform_data_path'")

    val (prefixes, schema, needEtl) =
      if (p1.nonEmpty) (etledPrefixes, etledSchema, false)
      else (rawPrefixes, rawSchema, true)

    // get train data paths
    val trainPaths = validPaths.filter(_.startsWith(prefixes.head))
    if (isToTrain) {
      require(trainPaths.nonEmpty, s"requires at least one path for train file." +
        s" Please specify it by '-dataPath=${prefixes(0)}your_train_data_path'")
    }

    // get eval path
    val evalPaths = validPaths.filter(_.startsWith(prefixes(1)))

    // get and check train data paths
    val transformPaths = validPaths.filter(_.startsWith(prefixes(2)))
    if (isToTransform) {
      require(transformPaths.nonEmpty, s"requires at least one path for transform file." +
        s" Please specify it by '-dataPath=${prefixes(2)}your_transform_data_path'")
    }

    // check data paths not specified type
    val unknownPaths = validPaths.filterNot(p => prefixes.exists(p.startsWith(_)))
    require(unknownPaths.isEmpty, s"Unknown type for data path: ${unknownPaths.head}, requires to specify" +
      s" the type for each data path by adding the prefix '${prefixes(0)}' or '${prefixes(1)}' or '${prefixes(2)}'.")

    (Array(trainPaths.map(_.stripPrefix(prefixes.head)),
      evalPaths.map(_.stripPrefix(prefixes(1))),
      transformPaths.map(_.stripPrefix(prefixes(2)))), schema, needEtl)
  }
}
