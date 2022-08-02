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
    substring_index(input_file_name(), ".", 1), "/", -1)
}

private object CsvReader {

  def readRaw(spark: SparkSession, paths: Seq[String], optionsMap: Map[String, String]): DataFrame = {

    val rawSchema = StructType(Array(
      StructField("reference_pool_id", StringType),
      StructField("loan_id", LongType),
      StructField("monthly_reporting_period", StringType),
      StructField("orig_channel", StringType),
      StructField("seller_name", StringType),
      StructField("servicer", StringType),
      StructField("master_servicer", StringType),
      StructField("orig_interest_rate", DoubleType),
      StructField("interest_rate", DoubleType),
      StructField("orig_upb", DoubleType),
      StructField("upb_at_issuance", StringType),
      StructField("current_actual_upb", DoubleType),
      StructField("orig_loan_term", IntegerType),
      StructField("orig_date", StringType),
      StructField("first_pay_date", StringType),    
      StructField("loan_age", DoubleType),
      StructField("remaining_months_to_legal_maturity", DoubleType),
      StructField("adj_remaining_months_to_maturity", DoubleType),
      StructField("maturity_date", StringType),
      StructField("orig_ltv", DoubleType),
      StructField("orig_cltv", DoubleType),
      StructField("num_borrowers", DoubleType),
      StructField("dti", DoubleType),
      StructField("borrower_credit_score", DoubleType),
      StructField("coborrow_credit_score", DoubleType),
      StructField("first_home_buyer", StringType),
      StructField("loan_purpose", StringType),
      StructField("property_type", StringType),
      StructField("num_units", IntegerType),
      StructField("occupancy_status", StringType),
      StructField("property_state", StringType),
      StructField("msa", DoubleType),
      StructField("zip", IntegerType),
      StructField("mortgage_insurance_percent", DoubleType),
      StructField("product_type", StringType),
      StructField("prepayment_penalty_indicator", StringType),
      StructField("interest_only_loan_indicator", StringType),
      StructField("interest_only_first_principal_and_interest_payment_date", StringType),
      StructField("months_to_amortization", StringType),
      StructField("current_loan_delinquency_status", IntegerType),
      StructField("loan_payment_history", StringType),
      StructField("mod_flag", StringType),
      StructField("mortgage_insurance_cancellation_indicator", StringType),
      StructField("zero_balance_code", StringType),
      StructField("zero_balance_effective_date", StringType),
      StructField("upb_at_the_time_of_removal", StringType),
      StructField("repurchase_date", StringType),
      StructField("scheduled_principal_current", StringType),
      StructField("total_principal_current", StringType),
      StructField("unscheduled_principal_current", StringType),
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
      StructField("original_list_start_date", StringType),
      StructField("original_list_price", StringType),
      StructField("current_list_start_date", StringType),
      StructField("current_list_price", StringType),
      StructField("borrower_credit_score_at_issuance", StringType),
      StructField("co-borrower_credit_score_at_issuance", StringType),
      StructField("borrower_credit_score_current", StringType),
      StructField("co-Borrower_credit_score_current", StringType),
      StructField("mortgage_insurance_type", DoubleType),
      StructField("servicing_activity_indicator", StringType),
      StructField("current_period_modification_loss_amount", StringType),
      StructField("cumulative_modification_loss_amount", StringType),
      StructField("current_period_credit_event_net_gain_or_loss", StringType),
      StructField("cumulative_credit_event_net_gain_or_loss", StringType),
      StructField("homeready_program_indicator", StringType),
      StructField("foreclosure_principal_write_off_amount", StringType),
      StructField("relocation_mortgage_indicator", StringType),
      StructField("zero_balance_code_change_date", StringType),
      StructField("loan_holdback_indicator", StringType),
      StructField("loan_holdback_effective_date", StringType),
      StructField("delinquent_accrued_interest", StringType),
      StructField("property_valuation_method", StringType),
      StructField("high_balance_loan_indicator", StringType),
      StructField("arm_initial_fixed-rate_period_lt_5_yr_indicator", StringType),
      StructField("arm_product_type", StringType),
      StructField("initial_fixed-rate_period", StringType),
      StructField("interest_rate_adjustment_frequency", StringType),
      StructField("next_interest_rate_adjustment_date", StringType),
      StructField("next_payment_change_date", StringType),
      StructField("index", StringType),
      StructField("arm_cap_structure", StringType),
      StructField("initial_interest_rate_cap_up_percent", StringType),
      StructField("periodic_interest_rate_cap_up_percent", StringType),
      StructField("lifetime_interest_rate_cap_up_percent", StringType),
      StructField("mortgage_margin", StringType),
      StructField("arm_balloon_indicator", StringType),
      StructField("arm_plan_number", StringType),
      StructField("borrower_assistance_plan", StringType),
      StructField("hltv_refinance_option_indicator", StringType),
      StructField("deal_name", StringType),
      StructField("repurchase_make_whole_proceeds_flag", StringType),
      StructField("alternative_delinquency_resolution", StringType),
      StructField("alternative_delinquency_resolution_count", StringType),
      StructField("total_deferral_amount", StringType)
      )
    )

    spark.read
      .options(optionsMap)
      .option("nullValue", "")
      .option("delimiter", "|")
      .schema(rawSchema)
      .csv(paths: _*)
      .withColumn("quarter", GetQuarterFromCsvFileName())
  }
}

object extractPerfColumns{
  def apply(rawDf : DataFrame) : DataFrame = {
    val perfDf = rawDf.select(
      col("loan_id"),
      date_format(to_date(col("monthly_reporting_period"),"MMyyyy"), "MM/dd/yyyy").as("monthly_reporting_period"),
      upper(col("servicer")).as("servicer"),
      col("interest_rate"),
      col("current_actual_upb"),
      col("loan_age"),
      col("remaining_months_to_legal_maturity"),
      col("adj_remaining_months_to_maturity"),
      date_format(to_date(col("maturity_date"),"MMyyyy"), "MM/yyyy").as("maturity_date"),
      col("msa"),
      col("current_loan_delinquency_status"),
      col("mod_flag"),
      col("zero_balance_code"),
      date_format(to_date(col("zero_balance_effective_date"),"MMyyyy"), "MM/yyyy").as("zero_balance_effective_date"),
      date_format(to_date(col("last_paid_installment_date"),"MMyyyy"), "MM/dd/yyyy").as("last_paid_installment_date"),
      date_format(to_date(col("foreclosed_after"),"MMyyyy"), "MM/dd/yyyy").as("foreclosed_after"),
      date_format(to_date(col("disposition_date"),"MMyyyy"), "MM/dd/yyyy").as("disposition_date"),
      col("foreclosure_costs"),
      col("prop_preservation_and_repair_costs"),
      col("asset_recovery_costs"),
      col("misc_holding_expenses"),
      col("holding_taxes"),
      col("net_sale_proceeds"),
      col("credit_enhancement_proceeds"),
      col("repurchase_make_whole_proceeds"),
      col("other_foreclosure_proceeds"),
      col("non_interest_bearing_upb"),
      col("principal_forgiveness_upb"),
      col("repurchase_make_whole_proceeds_flag"),
      col("foreclosure_principal_write_off_amount"),
      col("servicing_activity_indicator"),
      col("quarter")
    )

    perfDf.select("*").filter("current_actual_upb != 0.0")
  }
}

object extractAcqColumns{
  def apply(rawDf : DataFrame) : DataFrame = {
    val acqDf = rawDf.select(
      col("loan_id"),
      col("orig_channel"),
      upper(col("seller_name")).as("seller_name"),
      col("orig_interest_rate"),
      col("orig_upb"),
      col("orig_loan_term"),
      date_format(to_date(col("orig_date"),"MMyyyy"), "MM/yyyy").as("orig_date"),
      date_format(to_date(col("first_pay_date"),"MMyyyy"), "MM/yyyy").as("first_pay_date"),
      col("orig_ltv"),
      col("orig_cltv"),
      col("num_borrowers"),
      col("dti"),
      col("borrower_credit_score"),
      col("first_home_buyer"),
      col("loan_purpose"),
      col("property_type"),
      col("num_units"),
      col("occupancy_status"),
      col("property_state"),
      col("zip"),
      col("mortgage_insurance_percent"),
      col("product_type"),
      col("coborrow_credit_score"),
      col("mortgage_insurance_type"),
      col("relocation_mortgage_indicator"),
      col("quarter"),
      dense_rank().over(Window.partitionBy("loan_id").orderBy(to_date(col("monthly_reporting_period"),"MMyyyy"))).as("rank")
    )

    acqDf.select("*").filter(col("rank") === 1).drop("rank")
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

  def csv(spark: SparkSession, dataPaths: Seq[String], hasHeader: Boolean): DataFrame = {
    val optionsMap = Map("header" -> hasHeader.toString)
    val rawDf_csv = CsvReader.readRaw(spark, dataPaths, optionsMap)
    
    val (dataPaths, outPath, tmpPath) = checkAndGetPaths(xgbArgs.dataPaths)
    rawDf_csv.write.mode("overwrite").parquet(tmpPath)
    val rawDf = spark.read.parquet(tmpPath)
    
    val perfDf = extractPerfColumns(rawDf)
    val acqDf = extractAcqColumns(rawDf)
    transform(
      perfDf,
      acqDf,
      spark
    )
  }

  def parquet(spark: SparkSession, dataPaths: Seq[String]): DataFrame = {
    val rawDf = spark.read.parquet(dataPaths: _*)
    val perfDf = extractPerfColumns(rawDf)
    val acqDf = extractAcqColumns(rawDf)
    transform(
      perfDf,
      acqDf,
      spark
    )
  }

  def orc(spark: SparkSession, dataPaths: Seq[String]): DataFrame = {
    val rawDf = spark.read.orc(dataPaths: _*)
    val perfDf = extractPerfColumns(rawDf)
    val acqDf = extractAcqColumns(rawDf)
    transform(
      perfDf,
      acqDf,
      spark
    )
  }
}
