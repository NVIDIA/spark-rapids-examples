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

package com.nvidia.spark.examples.mortgage

import org.apache.spark.sql.types.{FloatType, IntegerType, StructField, StructType, DoubleType}

private[mortgage] trait Mortgage {
  val appName = "Mortgage"
  val labelColName = "delinquency_12"

  protected val categaryCols = List(
    ("orig_channel", FloatType),
    ("first_home_buyer", FloatType),
    ("loan_purpose", FloatType),
    ("property_type", FloatType),
    ("occupancy_status", FloatType),
    ("property_state", FloatType),
    ("product_type", FloatType),
    ("relocation_mortgage_indicator", FloatType),
    ("seller_name", FloatType),
    ("mod_flag", FloatType)
  )

  protected val numericCols = List(
    ("orig_interest_rate", FloatType),
    ("orig_upb", DoubleType),
    ("orig_loan_term", IntegerType),
    ("orig_ltv", FloatType),
    ("orig_cltv", FloatType),
    ("num_borrowers", FloatType),
    ("dti", FloatType),
    ("borrower_credit_score", FloatType),
    ("num_units", IntegerType),
    ("zip", IntegerType),
    ("mortgage_insurance_percent", FloatType),
    ("current_loan_delinquency_status", IntegerType),
    ("current_actual_upb", FloatType),
    ("interest_rate", FloatType),
    ("loan_age", FloatType),
    ("msa", FloatType),
    ("non_interest_bearing_upb", FloatType),
    (labelColName, IntegerType)
  )

  lazy val schema = StructType((categaryCols ++ numericCols).map(col => StructField(col._1, col._2)))
  lazy val featureNames = schema.filter(_.name != labelColName).map(_.name).toArray

  val commParamMap = Map(
    "objective" -> "binary:logistic",
    "num_round" -> 100)
}
