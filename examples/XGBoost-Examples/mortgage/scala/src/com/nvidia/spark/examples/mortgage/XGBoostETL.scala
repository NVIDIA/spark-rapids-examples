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

import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql.{Column, DataFrame, SparkSession}


object GetQuarterFromCsvFileName {
  // The format is path/TYPE_yyyy\QQ.txt followed by a (_index)* where index is a single digit number [0-9]
  // i.e. mortgage/perf/Performance_2003Q4.txt_0_1
  // So we strip off the .txt and everything after it
  // and then take everything after the last remaining _
  def apply(): Column = substring_index(
    substring_index(input_file_name(), ".", 1), "_", -1)
}

private object CsvReader {

  def readPerformance(spark: SparkSession, paths: Seq[String], optionsMap: Map[String, String]): DataFrame = {
    val performanceSchema = StructType(Array(
      StructField("loan_id", LongType),
      StructField("monthly_reporting_period", StringType),
      StructField("servicer", StringType),
      StructField("interest_rate", DoubleType),
      StructField("current_actual_upb", DoubleType),
      StructField("loan_age", DoubleType),
      StructField("remaining_months_to_legal_maturity", DoubleType),
      StructField("adj_remaining_months_to_maturity", DoubleType),
      StructField("maturity_date", StringType),
      StructField("msa", DoubleType),
      StructField("current_loan_delinquency_status", IntegerType),
      StructField("mod_flag", StringType),
      StructField("zero_balance_code", StringType),
      StructField("zero_balance_effective_date", StringType),
      StructField("last_paid_installment_date", StringType),
      StructField("foreclosed_after", StringType),
      StructField("disposition_date", StringType),
      StructField("foreclosure_costs", DoubleType),
      StructField("prop_preservation_and_repair_costs", DoubleType),
      StructField("asset_recovery_costs", DoubleType),
      StructField("misc_holding_expenses", DoubleType),
      StructField("holding_taxes", DoubleType),
      StructField("net_sale_proceeds", DoubleType),
      StructField("credit_enhancement_proceeds", DoubleType),
      StructField("repurchase_make_whole_proceeds", StringType),
      StructField("other_foreclosure_proceeds", DoubleType),
      StructField("non_interest_bearing_upb", DoubleType),
      StructField("principal_forgiveness_upb", StringType),
      StructField("repurchase_make_whole_proceeds_flag", StringType),
      StructField("foreclosure_principal_write_off_amount", StringType),
      StructField("servicing_activity_indicator", StringType))
    )

    spark.read
      .options(optionsMap)
      .option("nullValue", "")
      .option("delimiter", "|")
      .option("parserLib", "univocity")
      .schema(performanceSchema)
      .csv(paths: _*)
      .withColumn("quarter", GetQuarterFromCsvFileName())
  }

  def readAcquisition(spark: SparkSession, paths: Seq[String], optionsMap: Map[String, String]): DataFrame = {
    val acquisitionSchema = StructType(Array(
      StructField("loan_id", LongType),
      StructField("orig_channel", StringType),
      StructField("seller_name", StringType),
      StructField("orig_interest_rate", DoubleType),
      StructField("orig_upb", IntegerType),
      StructField("orig_loan_term", IntegerType),
      StructField("orig_date", StringType),
      StructField("first_pay_date", StringType),
      StructField("orig_ltv", DoubleType),
      StructField("orig_cltv", DoubleType),
      StructField("num_borrowers", DoubleType),
      StructField("dti", DoubleType),
      StructField("borrower_credit_score", DoubleType),
      StructField("first_home_buyer", StringType),
      StructField("loan_purpose", StringType),
      StructField("property_type", StringType),
      StructField("num_units", IntegerType),
      StructField("occupancy_status", StringType),
      StructField("property_state", StringType),
      StructField("zip", IntegerType),
      StructField("mortgage_insurance_percent", DoubleType),
      StructField("product_type", StringType),
      StructField("coborrow_credit_score", DoubleType),
      StructField("mortgage_insurance_type", DoubleType),
      StructField("relocation_mortgage_indicator", StringType))
    )

    spark.read
      .options(optionsMap)
      .option("delimiter", "|")
      .schema(acquisitionSchema)
      .csv(paths: _*)
      .withColumn("quarter", GetQuarterFromCsvFileName())
  }
}

object NameMapping {
  /**
    * Returns a dataframe with two columns named based off of the column names passed in.
    * The fromColName has the original name we want to clean up, the toColName
    * will have the name we want to go to, the unambiguous name.
    */
  def apply(spark: SparkSession, fromColName: String, toColName: String): DataFrame = {
    import spark.sqlContext.implicits._
    broadcast(Seq(
      ("WITMER FUNDING, LLC", "Witmer"),
      ("WELLS FARGO CREDIT RISK TRANSFER SECURITIES TRUST 2015", "Wells Fargo"),
      ("WELLS FARGO BANK,  NA" , "Wells Fargo"),
      ("WELLS FARGO BANK, N.A." , "Wells Fargo"),
      ("WELLS FARGO BANK, NA" , "Wells Fargo"),
      ("USAA FEDERAL SAVINGS BANK" , "USAA"),
      ("UNITED SHORE FINANCIAL SERVICES, LLC D\\/B\\/A UNITED WHOLESALE MORTGAGE" , "United Seq(e"),
      ("U.S. BANK N.A." , "US Bank"),
      ("SUNTRUST MORTGAGE INC." , "Suntrust"),
      ("STONEGATE MORTGAGE CORPORATION" , "Stonegate Mortgage"),
      ("STEARNS LENDING, LLC" , "Stearns Lending"),
      ("STEARNS LENDING, INC." , "Stearns Lending"),
      ("SIERRA PACIFIC MORTGAGE COMPANY, INC." , "Sierra Pacific Mortgage"),
      ("REGIONS BANK" , "Regions"),
      ("RBC MORTGAGE COMPANY" , "RBC"),
      ("QUICKEN LOANS INC." , "Quicken Loans"),
      ("PULTE MORTGAGE, L.L.C." , "Pulte Mortgage"),
      ("PROVIDENT FUNDING ASSOCIATES, L.P." , "Provident Funding"),
      ("PROSPECT MORTGAGE, LLC" , "Prospect Mortgage"),
      ("PRINCIPAL RESIDENTIAL MORTGAGE CAPITAL RESOURCES, LLC" , "Principal Residential"),
      ("PNC BANK, N.A." , "PNC"),
      ("PMT CREDIT RISK TRANSFER TRUST 2015-2" , "PennyMac"),
      ("PHH MORTGAGE CORPORATION" , "PHH Mortgage"),
      ("PENNYMAC CORP." , "PennyMac"),
      ("PACIFIC UNION FINANCIAL, LLC" , "Other"),
      ("OTHER" , "Other"),
      ("NYCB MORTGAGE COMPANY, LLC" , "NYCB"),
      ("NEW YORK COMMUNITY BANK" , "NYCB"),
      ("NETBANK FUNDING SERVICES" , "Netbank"),
      ("NATIONSTAR MORTGAGE, LLC" , "Nationstar Mortgage"),
      ("METLIFE BANK, NA" , "Metlife"),
      ("LOANDEPOT.COM, LLC" , "LoanDepot.com"),
      ("J.P. MORGAN MADISON AVENUE SECURITIES TRUST, SERIES 2015-1" , "JP Morgan Chase"),
      ("J.P. MORGAN MADISON AVENUE SECURITIES TRUST, SERIES 2014-1" , "JP Morgan Chase"),
      ("JPMORGAN CHASE BANK, NATIONAL ASSOCIATION" , "JP Morgan Chase"),
      ("JPMORGAN CHASE BANK, NA" , "JP Morgan Chase"),
      ("JP MORGAN CHASE BANK, NA" , "JP Morgan Chase"),
      ("IRWIN MORTGAGE, CORPORATION" , "Irwin Mortgage"),
      ("IMPAC MORTGAGE CORP." , "Impac Mortgage"),
      ("HSBC BANK USA, NATIONAL ASSOCIATION" , "HSBC"),
      ("HOMEWARD RESIDENTIAL, INC." , "Homeward Mortgage"),
      ("HOMESTREET BANK" , "Other"),
      ("HOMEBRIDGE FINANCIAL SERVICES, INC." , "HomeBridge"),
      ("HARWOOD STREET FUNDING I, LLC" , "Harwood Mortgage"),
      ("GUILD MORTGAGE COMPANY" , "Guild Mortgage"),
      ("GMAC MORTGAGE, LLC (USAA FEDERAL SAVINGS BANK)" , "GMAC"),
      ("GMAC MORTGAGE, LLC" , "GMAC"),
      ("GMAC (USAA)" , "GMAC"),
      ("FREMONT BANK" , "Fremont Bank"),
      ("FREEDOM MORTGAGE CORP." , "Freedom Mortgage"),
      ("FRANKLIN AMERICAN MORTGAGE COMPANY" , "Franklin America"),
      ("FLEET NATIONAL BANK" , "Fleet National"),
      ("FLAGSTAR CAPITAL MARKETS CORPORATION" , "Flagstar Bank"),
      ("FLAGSTAR BANK, FSB" , "Flagstar Bank"),
      ("FIRST TENNESSEE BANK NATIONAL ASSOCIATION" , "Other"),
      ("FIFTH THIRD BANK" , "Fifth Third Bank"),
      ("FEDERAL HOME LOAN BANK OF CHICAGO" , "Fedral Home of Chicago"),
      ("FDIC, RECEIVER, INDYMAC FEDERAL BANK FSB" , "FDIC"),
      ("DOWNEY SAVINGS AND LOAN ASSOCIATION, F.A." , "Downey Mortgage"),
      ("DITECH FINANCIAL LLC" , "Ditech"),
      ("CITIMORTGAGE, INC." , "Citi"),
      ("CHICAGO MORTGAGE SOLUTIONS DBA INTERFIRST MORTGAGE COMPANY" , "Chicago Mortgage"),
      ("CHICAGO MORTGAGE SOLUTIONS DBA INTERBANK MORTGAGE COMPANY" , "Chicago Mortgage"),
      ("CHASE HOME FINANCE, LLC" , "JP Morgan Chase"),
      ("CHASE HOME FINANCE FRANKLIN AMERICAN MORTGAGE COMPANY" , "JP Morgan Chase"),
      ("CHASE HOME FINANCE (CIE 1)" , "JP Morgan Chase"),
      ("CHASE HOME FINANCE" , "JP Morgan Chase"),
      ("CASHCALL, INC." , "CashCall"),
      ("CAPITAL ONE, NATIONAL ASSOCIATION" , "Capital One"),
      ("CALIBER HOME LOANS, INC." , "Caliber Funding"),
      ("BISHOPS GATE RESIDENTIAL MORTGAGE TRUST" , "Bishops Gate Mortgage"),
      ("BANK OF AMERICA, N.A." , "Bank of America"),
      ("AMTRUST BANK" , "AmTrust"),
      ("AMERISAVE MORTGAGE CORPORATION" , "Amerisave"),
      ("AMERIHOME MORTGAGE COMPANY, LLC" , "AmeriHome Mortgage"),
      ("ALLY BANK" , "Ally Bank"),
      ("ACADEMY MORTGAGE CORPORATION" , "Academy Mortgage"),
      ("NO CASH-OUT REFINANCE" , "OTHER REFINANCE"),
      ("REFINANCE - NOT SPECIFIED" , "OTHER REFINANCE"),
      ("Other REFINANCE" , "OTHER REFINANCE")
    ).toDF(fromColName, toColName))
  }
}

private trait MortgageETL {
  var dataFrame: DataFrame = _

  def from(df: DataFrame): this.type = {
    dataFrame = df
    this
  }
}

private object PerformanceETL extends MortgageETL {

  def prepare: this.type = {
    dataFrame = dataFrame
      .withColumn("monthly_reporting_period", to_date(col("monthly_reporting_period"), "MM/dd/yyyy"))
      .withColumn("monthly_reporting_period_month", month(col("monthly_reporting_period")))
      .withColumn("monthly_reporting_period_year", year(col("monthly_reporting_period")))
      .withColumn("monthly_reporting_period_day", dayofmonth(col("monthly_reporting_period")))
      .withColumn("last_paid_installment_date", to_date(col("last_paid_installment_date"), "MM/dd/yyyy"))
      .withColumn("foreclosed_after", to_date(col("foreclosed_after"), "MM/dd/yyyy"))
      .withColumn("disposition_date", to_date(col("disposition_date"), "MM/dd/yyyy"))
      .withColumn("maturity_date", to_date(col("maturity_date"), "MM/yyyy"))
      .withColumn("zero_balance_effective_date", to_date(col("zero_balance_effective_date"), "MM/yyyy"))
      .withColumn("current_actual_upb", col("current_actual_upb"))
      .withColumn("current_loan_delinquency_status", col("current_loan_delinquency_status"))
    this
  }

  def createDelinquency(spark: SparkSession): this.type = {
    val aggDF = dataFrame
      .select(
        col("quarter"),
        col("loan_id"),
        col("current_loan_delinquency_status"),
        when(col("current_loan_delinquency_status") >= 1, col("monthly_reporting_period")).alias("delinquency_30"),
        when(col("current_loan_delinquency_status") >= 3, col("monthly_reporting_period")).alias("delinquency_90"),
        when(col("current_loan_delinquency_status") >= 6, col("monthly_reporting_period")).alias("delinquency_180")
      )
      .groupBy("quarter", "loan_id")
      .agg(
        max("current_loan_delinquency_status").alias("delinquency_12"),
        min("delinquency_30").alias("delinquency_30"),
        min("delinquency_90").alias("delinquency_90"),
        min("delinquency_180").alias("delinquency_180")
      )
      .select(
        col("quarter"),
        col("loan_id"),
        (col("delinquency_12") >= 1).alias("ever_30"),
        (col("delinquency_12") >= 3).alias("ever_90"),
        (col("delinquency_12") >= 6).alias("ever_180"),
        col("delinquency_30"),
        col("delinquency_90"),
        col("delinquency_180")
      )

    val joinedDf = dataFrame
      .withColumnRenamed("monthly_reporting_period", "timestamp")
      .withColumnRenamed("monthly_reporting_period_month", "timestamp_month")
      .withColumnRenamed("monthly_reporting_period_year", "timestamp_year")
      .withColumnRenamed("current_loan_delinquency_status", "delinquency_12")
      .withColumnRenamed("current_actual_upb", "upb_12")
      .select("quarter", "loan_id", "timestamp", "delinquency_12", "upb_12", "timestamp_month", "timestamp_year")
      .join(aggDF, Seq("loan_id", "quarter"), "left_outer")

    // calculate the 12 month delinquency and upb values
    val months = 12
    val monthArray = 0.until(months).toArray
    val testDf = joinedDf
      // explode on a small amount of data is actually slightly more efficient than a cross join
      .withColumn("month_y", explode(lit(monthArray)))
      .select(
        col("quarter"),
        floor(((col("timestamp_year") * 12 + col("timestamp_month")) - 24000) / months).alias("josh_mody"),
        floor(((col("timestamp_year") * 12 + col("timestamp_month")) - 24000 - col("month_y")) / months).alias("josh_mody_n"),
        col("ever_30"),
        col("ever_90"),
        col("ever_180"),
        col("delinquency_30"),
        col("delinquency_90"),
        col("delinquency_180"),
        col("loan_id"),
        col("month_y"),
        col("delinquency_12"),
        col("upb_12")
      )
      .groupBy("quarter", "loan_id", "josh_mody_n", "ever_30", "ever_90", "ever_180", "delinquency_30", "delinquency_90", "delinquency_180", "month_y")
      .agg(max("delinquency_12").alias("delinquency_12"), min("upb_12").alias("upb_12"))
      .withColumn("timestamp_year", floor((lit(24000) + (col("josh_mody_n") * lit(months)) + (col("month_y") - 1)) / lit(12)))
      .withColumn("timestamp_month_tmp", pmod(lit(24000) + (col("josh_mody_n") * lit(months)) + col("month_y"), lit(12)))
      .withColumn("timestamp_month", when(col("timestamp_month_tmp") === lit(0), lit(12)).otherwise(col("timestamp_month_tmp")))
      .withColumn("delinquency_12", ((col("delinquency_12") > 3).cast("int") + (col("upb_12") === 0).cast("int")).alias("delinquency_12"))
      .drop("timestamp_month_tmp", "josh_mody_n", "month_y")

    dataFrame = dataFrame
      .withColumnRenamed("monthly_reporting_period_month", "timestamp_month")
      .withColumnRenamed("monthly_reporting_period_year", "timestamp_year")
      .join(testDf, Seq("quarter", "loan_id", "timestamp_year", "timestamp_month"), "left").drop("timestamp_year", "timestamp_month")
    this
  }
}

private object AcquisitionETL extends MortgageETL {

  def createAcquisition(spark: SparkSession): this.type = {
    val nameMapping = NameMapping(spark, "from_seller_name", "to_seller_name")
    dataFrame = dataFrame
      .join(nameMapping, col("seller_name") === col("from_seller_name"), "left")
      .drop("from_seller_name")
      /* backup the original name before we replace it */
      .withColumn("old_name", col("seller_name"))
      /* replace seller_name with the new version if we found one in the mapping, or the old version
       if we didn't */
      .withColumn("seller_name", coalesce(col("to_seller_name"), col("seller_name")))
      .drop("to_seller_name")
      .withColumn("orig_date", to_date(col("orig_date"), "MM/yyyy"))
      .withColumn("first_pay_date", to_date(col("first_pay_date"), "MM/yyyy"))
    this
  }

  def cleanPrime(perfDF: DataFrame): this.type = {
    dataFrame = perfDF.join(dataFrame, Seq("loan_id", "quarter"), "inner").drop("quarter")
    this
  }
}

object XGBoostETL extends Mortgage {

  private lazy val allCols = (categaryCols ++ numericCols).map(c => col(c._1))
  private var cachedDictDF: DataFrame = _

  /**
    * Generate a dictionary from string to numeric value for multiple category columns.
    *
    * (Copied the solution of casting string to numeric from the utils of DLRM.)
    */
  private def genDictionary(etlDF: DataFrame, colNames: Seq[String]): DataFrame = {
    val cntTable = etlDF
      .select(posexplode(array(colNames.map(col(_)): _*)))
      .withColumnRenamed("pos", "column_id")
      .withColumnRenamed("col", "data")
      .filter("data is not null")
      .groupBy("column_id", "data")
      .count()
    val windowed = Window.partitionBy("column_id").orderBy(desc("count"))
    cntTable
      .withColumn("id", row_number().over(windowed))
      .drop("count")
  }

  /**
    * Cast all the category columns to numeric columns in the given data frame.
    * Then it is suitable for XGBoost training/transforming
    */
  private def castStringColumnsToNumeric(inputDF: DataFrame, spark: SparkSession): DataFrame = {
    val cateColNames = categaryCols.map(_._1)
    cachedDictDF = genDictionary(inputDF, cateColNames).cache()

    // Generate the final table with all columns being numeric.
    cateColNames.foldLeft(inputDF) {
      case (df, colName) =>
        val colPos = cateColNames.indexOf(colName)
        val colDictDF = cachedDictDF
          .filter(col("column_id") === colPos)
          .drop("column_id")
          .withColumnRenamed("data", colName)
        df.join(broadcast(colDictDF), Seq(colName), "left")
          .drop(colName)
          .withColumnRenamed("id", colName)
    }
  }

  private def transform(perfDF: DataFrame, acqDF: DataFrame, spark: SparkSession): DataFrame = {
    val etlPerfDF = PerformanceETL.from(perfDF)
      .prepare
      .createDelinquency(spark)
      .dataFrame
    val cleanDF = AcquisitionETL.from(acqDF)
      .createAcquisition(spark)
      .cleanPrime(etlPerfDF)
      .dataFrame

    // Convert to xgb required Dataset
    castStringColumnsToNumeric(cleanDF, spark)
      .select(allCols: _*)
      .withColumn(labelColName, when(col(labelColName) > 0, 1).otherwise(0))
      .na.fill(0.0f)
  }

  def clean(): Unit = {
    if (cachedDictDF != null) {
      cachedDictDF.unpersist()
      cachedDictDF = null
    }
  }

  def saveDictTable(outPath: String): Unit = {
    if (cachedDictDF != null) {
      // The dict data is small, so merge it into one file.
      cachedDictDF
        .repartition(1)
        .write
        .mode("overwrite")
        .parquet(outPath)
    }
  }

  def csv(spark: SparkSession, perfPaths: Seq[String], acqPaths: Seq[String], hasHeader: Boolean): DataFrame = {
    val optionsMap = Map("header" -> hasHeader.toString)
    transform(
      CsvReader.readPerformance(spark, perfPaths, optionsMap),
      CsvReader.readAcquisition(spark, acqPaths, optionsMap),
      spark
    )
  }

  def parquet(spark: SparkSession, perfPaths: Seq[String], acqPaths: Seq[String]): DataFrame = {
    transform(
      spark.read.parquet(perfPaths: _*),
      spark.read.parquet(acqPaths: _*),
      spark
    )
  }

  def orc(spark: SparkSession, perfPaths: Seq[String], acqPaths: Seq[String]): DataFrame = {
    transform(
      spark.read.orc(perfPaths: _*),
      spark.read.orc(acqPaths: _*),
      spark
    )
  }
}
