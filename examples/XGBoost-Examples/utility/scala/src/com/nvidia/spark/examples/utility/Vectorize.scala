
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

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.FloatType

object Vectorize {
  def apply(df: DataFrame, labelName: String, changeLabelName: Boolean = true): DataFrame = {
    val features = df.schema.collect { case f if f.name != labelName => f.name }
    val toFloat = df.schema.map(f => col(f.name).cast(FloatType))
    val labelCol = if (changeLabelName) col(labelName).alias("label") else col(labelName)
    new VectorAssembler()
      .setInputCols(features.toArray)
      .setOutputCol("features")
      .transform(df.select(toFloat: _*))
      .select(col("features"), labelCol)
  }

  def apply(df: DataFrame, featureNames: Seq[String], labelName: String): DataFrame = {
    val toFloat = df.schema.map(f => col(f.name).cast(FloatType))
    new VectorAssembler()
      .setInputCols(featureNames.toArray)
      .setOutputCol("features")
      .transform(df.select(toFloat: _*))
      .select(col("features"), col(labelName))
  }

  def apply(featureNames: Seq[String], df: DataFrame, otherNames: String*): DataFrame = {
    val resultCols = (otherNames :+ "features").map(col(_))
    new VectorAssembler()
      .setInputCols(featureNames.toArray)
      .setOutputCol("features")
      .transform(df)
      .select(resultCols: _*)
  }

  def criteoApply(df: DataFrame, featureNames: Seq[String], labelName: String): DataFrame = {
    val toFloat = df.schema.map(f => col(f.name).cast(FloatType))
    new VectorAssembler()
      .setHandleInvalid("keep")
      .setInputCols(featureNames.toArray)
      .setOutputCol("features")
      .transform(df.select(toFloat: _*))
      .select(col("features"), col(labelName))
  }

}
