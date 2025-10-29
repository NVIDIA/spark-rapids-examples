#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# -----
#
# Certain portions of the contents of this file are derived from TPC-DS version 3.2.0
# (retrieved from www.tpc.org/tpc_documents_current_versions/current_specifications5.asp).
# Such portions are subject to copyrights held by Transaction Processing Performance Council (“TPC”)
# and licensed under the TPC EULA (a copy of which accompanies this file as “TPC EULA” and is also
# available at http://www.tpc.org/tpc_documents_current_versions/current_specifications5.asp) (the “TPC EULA”).
#
# You may not use this file except in compliance with the TPC EULA.
# DISCLAIMER: Portions of this file is derived from the TPC-DS Benchmark and as such any results
# obtained using this file are not comparable to published TPC-DS Benchmark results, as the results
# obtained from using this file do not comply with the TPC-DS Benchmark.
#

import pyspark
from pyspark.sql.types import *

if not hasattr(pyspark.sql.types, "VarcharType"):
    # this is a version of Spark that doesn't have fixed- and max-length string types
    setattr(pyspark.sql.types, "VarcharType", lambda x: StringType())
    setattr(pyspark.sql.types, "CharType", lambda x: StringType())

from pyspark.sql.types import VarcharType, CharType

def decimalType(use_decimal, precision, scale):
    if use_decimal:
        return DecimalType(precision, scale)
    else:
        return DoubleType()

def get_schemas(use_decimal):
    """get the schemas of all tables. If use_decimal is True, DecimalType are applied, otherwide,
    DoubleType will be used for DecimalType.

    Args:
        use_decimal (bool): use decimal or not

    Returns:
        dict: {table_name: schema}
    """
    SCHEMAS = {}

    # The specification states that "Identifier means that the column shall be able to hold any
    # key value generated for that column". Some tables have more rows than others so we can
    # choose to use different types per table.
    identifier_int = IntegerType()
    identifier_long = LongType()

    SCHEMAS["customer_address"] = StructType([
        StructField("ca_address_sk", identifier_int, nullable=False),
        StructField("ca_address_id", CharType(16), nullable=False),
        StructField("ca_street_number", CharType(10)),
        StructField("ca_street_name", VarcharType(60)),
        StructField("ca_street_type", CharType(15)),
        StructField("ca_suite_number", CharType(10)),
        StructField("ca_city", VarcharType(60)),
        StructField("ca_county", VarcharType(30)),
        StructField("ca_state", CharType(2)),
        StructField("ca_zip", CharType(10)),
        StructField("ca_country", VarcharType(20)),
        StructField("ca_gmt_offset", decimalType(use_decimal, 5, 2)),
        StructField("ca_location_type", CharType(20))
    ])

    SCHEMAS["customer_demographics"] = StructType([
        StructField("cd_demo_sk", identifier_int, nullable=False),
        StructField("cd_gender", CharType(1)),
        StructField("cd_marital_status", CharType(1)),
        StructField("cd_education_status", CharType(20)),
        StructField("cd_purchase_estimate", LongType()),
        StructField("cd_credit_rating", CharType(10)),
        StructField("cd_dep_count", LongType()),
        StructField("cd_dep_employed_count", LongType()),
        StructField("cd_dep_college_count", LongType())
    ])

    SCHEMAS["date_dim"] = StructType([
        StructField("d_date_sk", identifier_int, nullable=False),
        StructField("d_date_id", CharType(16), nullable=False),
        StructField("d_date", DateType()),
        StructField("d_month_seq", LongType()),
        StructField("d_week_seq", LongType()),
        StructField("d_quarter_seq", LongType()),
        StructField("d_year", LongType()),
        StructField("d_dow", LongType()),
        StructField("d_moy", LongType()),
        StructField("d_dom", LongType()),
        StructField("d_qoy", LongType()),
        StructField("d_fy_year", LongType()),
        StructField("d_fy_quarter_seq", LongType()),
        StructField("d_fy_week_seq", LongType()),
        StructField("d_day_name", CharType(9)),
        StructField("d_quarter_name", CharType(6)),
        StructField("d_holiday", CharType(1)),
        StructField("d_weekend", CharType(1)),
        StructField("d_following_holiday", CharType(1)),
        StructField("d_first_dom", LongType()),
        StructField("d_last_dom", LongType()),
        StructField("d_same_day_ly", LongType()),
        StructField("d_same_day_lq", LongType()),
        StructField("d_current_day", CharType(1)),
        StructField("d_current_week", CharType(1)),
        StructField("d_current_month", CharType(1)),
        StructField("d_current_quarter", CharType(1)),
        StructField("d_current_year", CharType(1))
    ])

    SCHEMAS["warehouse"] = StructType([
        StructField("w_warehouse_sk", identifier_int, nullable=False),
        StructField("w_warehouse_id", CharType(16), nullable=False),
        StructField("w_warehouse_name", VarcharType(20)),
        StructField("w_warehouse_sq_ft", LongType()),
        StructField("w_street_number", CharType(10)),
        StructField("w_street_name", VarcharType(60)),
        StructField("w_street_type", CharType(15)),
        StructField("w_suite_number", CharType(10)),
        StructField("w_city", VarcharType(60)),
        StructField("w_county", VarcharType(30)),
        StructField("w_state", CharType(2)),
        StructField("w_zip", CharType(10)),
        StructField("w_country", VarcharType(20)),
        StructField("w_gmt_offset", decimalType(use_decimal, 5, 2))
    ])

    SCHEMAS["ship_mode"] = StructType([
        StructField("sm_ship_mode_sk", identifier_int, nullable=False),
        StructField("sm_ship_mode_id", CharType(16), nullable=False),
        StructField("sm_type", CharType(30)),
        StructField("sm_code", CharType(10)),
        StructField("sm_carrier", CharType(20)),
        StructField("sm_contract", CharType(20))
    ])

    SCHEMAS["time_dim"] = StructType([
        StructField("t_time_sk", identifier_int, nullable=False),
        StructField("t_time_id", CharType(16), nullable=False),
        StructField("t_time", LongType(), nullable=False),
        StructField("t_hour", LongType()),
        StructField("t_minute", LongType()),
        StructField("t_second", LongType()),
        StructField("t_am_pm", CharType(2)),
        StructField("t_shift", CharType(20)),
        StructField("t_sub_shift", CharType(20)),
        StructField("t_meal_time", CharType(20))
    ])

    SCHEMAS["reason"] = StructType([
        StructField("r_reason_sk", identifier_int, nullable=False),
        StructField("r_reason_id", CharType(16), nullable=False),
        StructField("r_reason_desc", CharType(100))
    ])

    SCHEMAS["income_band"] = StructType([
        StructField("ib_income_band_sk", identifier_int, nullable=False),
        StructField("ib_lower_bound", LongType()),
        StructField("ib_upper_bound", LongType())
    ])

    SCHEMAS["item"] = StructType([
        StructField("i_item_sk", identifier_int, nullable=False),
        StructField("i_item_id", CharType(16), nullable=False),
        StructField("i_rec_start_date", DateType()),
        StructField("i_rec_end_date", DateType()),
        StructField("i_item_desc", VarcharType(200)),
        StructField("i_current_price", decimalType(use_decimal, 7, 2)),
        StructField("i_wholesale_cost", decimalType(use_decimal, 7, 2)),
        StructField("i_brand_id", LongType()),
        StructField("i_brand", CharType(50)),
        StructField("i_class_id", LongType()),
        StructField("i_class", CharType(50)),
        StructField("i_category_id", LongType()),
        StructField("i_category", CharType(50)),
        StructField("i_manufact_id", LongType()),
        StructField("i_manufact", CharType(50)),
        StructField("i_size", CharType(20)),
        StructField("i_formulation", CharType(20)),
        StructField("i_color", CharType(20)),
        StructField("i_units", CharType(10)),
        StructField("i_container", CharType(10)),
        StructField("i_manager_id", LongType()),
        StructField("i_product_name", CharType(50))
    ])

    SCHEMAS["store"] = StructType([
        StructField("s_store_sk", identifier_int, nullable=False),
        StructField("s_store_id", CharType(16), nullable=False),
        StructField("s_rec_start_date", DateType()),
        StructField("s_rec_end_date", DateType()),
        StructField("s_closed_date_sk", identifier_int),
        StructField("s_store_name", VarcharType(50)),
        StructField("s_number_employees", LongType()),
        StructField("s_floor_space", LongType()),
        StructField("s_hours", CharType(20)),
        StructField("s_manager", VarcharType(40)),
        StructField("s_market_id", LongType()),
        StructField("s_geography_class", VarcharType(100)),
        StructField("s_market_desc", VarcharType(100)),
        StructField("s_market_manager", VarcharType(40)),
        StructField("s_division_id", LongType()),
        StructField("s_division_name", VarcharType(50)),
        StructField("s_company_id", LongType()),
        StructField("s_company_name", VarcharType(50)),
        StructField("s_street_number", VarcharType(10)),
        StructField("s_street_name", VarcharType(60)),
        StructField("s_street_type", CharType(15)),
        StructField("s_suite_number", CharType(10)),
        StructField("s_city", VarcharType(60)),
        StructField("s_county", VarcharType(30)),
        StructField("s_state", CharType(2)),
        StructField("s_zip", CharType(10)),
        StructField("s_country", VarcharType(20)),
        StructField("s_gmt_offset", decimalType(use_decimal, 5, 2)),
        StructField("s_tax_precentage", decimalType(use_decimal, 5, 2))
    ])

    SCHEMAS["call_center"] = StructType([
        StructField("cc_call_center_sk", identifier_int, nullable=False),
        StructField("cc_call_center_id", CharType(16), nullable=False),
        StructField("cc_rec_start_date", DateType()),
        StructField("cc_rec_end_date", DateType()),
        StructField("cc_closed_date_sk", identifier_int),
        StructField("cc_open_date_sk", identifier_int),
        StructField("cc_name", VarcharType(50)),
        StructField("cc_class", VarcharType(50)),
        StructField("cc_employees", LongType()),
        StructField("cc_sq_ft", LongType()),
        StructField("cc_hours", CharType(20)),
        StructField("cc_manager", VarcharType(40)),
        StructField("cc_mkt_id", LongType()),
        StructField("cc_mkt_class", CharType(50)),
        StructField("cc_mkt_desc", VarcharType(100)),
        StructField("cc_market_manager", VarcharType(40)),
        StructField("cc_division", LongType()),
        StructField("cc_division_name", VarcharType(50)),
        StructField("cc_company", LongType()),
        StructField("cc_company_name", CharType(50)),
        StructField("cc_street_number", CharType(10)),
        StructField("cc_street_name", VarcharType(60)),
        StructField("cc_street_type", CharType(15)),
        StructField("cc_suite_number", CharType(10)),
        StructField("cc_city", VarcharType(60)),
        StructField("cc_county", VarcharType(30)),
        StructField("cc_state", CharType(2)),
        StructField("cc_zip", CharType(10)),
        StructField("cc_country", VarcharType(20)),
        StructField("cc_gmt_offset", decimalType(use_decimal, 5, 2)),
        StructField("cc_tax_percentage", decimalType(use_decimal, 5, 2))
    ])

    SCHEMAS["customer"] = StructType([
        StructField("c_customer_sk", identifier_int, nullable=False),
        StructField("c_customer_id", CharType(16), nullable=False),
        StructField("c_current_cdemo_sk", identifier_int),
        StructField("c_current_hdemo_sk", identifier_int),
        StructField("c_current_addr_sk", identifier_int),
        StructField("c_first_shipto_date_sk", identifier_int),
        StructField("c_first_sales_date_sk", identifier_int),
        StructField("c_salutation", CharType(10)),
        StructField("c_first_name", CharType(20)),
        StructField("c_last_name", CharType(30)),
        StructField("c_preferred_cust_flag", CharType(1)),
        StructField("c_birth_day", LongType()),
        StructField("c_birth_month", LongType()),
        StructField("c_birth_year", LongType()),
        StructField("c_birth_country", VarcharType(20)),
        StructField("c_login", CharType(13)),
        StructField("c_email_address", CharType(50)),
        StructField("c_last_review_date_sk", identifier_int)
    ])

    SCHEMAS["web_site"] = StructType([
        StructField("web_site_sk", identifier_int, nullable=False),
        StructField("web_site_id", CharType(16), nullable=False),
        StructField("web_rec_start_date", DateType()),
        StructField("web_rec_end_date", DateType()),
        StructField("web_name", VarcharType(50)),
        StructField("web_open_date_sk", identifier_int),
        StructField("web_close_date_sk", identifier_int),
        StructField("web_class", VarcharType(50)),
        StructField("web_manager", VarcharType(40)),
        StructField("web_mkt_id", LongType()),
        StructField("web_mkt_class", VarcharType(50)),
        StructField("web_mkt_desc", VarcharType(100)),
        StructField("web_market_manager", VarcharType(40)),
        StructField("web_company_id", LongType()),
        StructField("web_company_name", CharType(50)),
        StructField("web_street_number", CharType(10)),
        StructField("web_street_name", VarcharType(60)),
        StructField("web_street_type", CharType(15)),
        StructField("web_suite_number", CharType(10)),
        StructField("web_city", VarcharType(60)),
        StructField("web_county", VarcharType(30)),
        StructField("web_state", CharType(2)),
        StructField("web_zip", CharType(10)),
        StructField("web_country", VarcharType(20)),
        StructField("web_gmt_offset", decimalType(use_decimal, 5, 2)),
        StructField("web_tax_percentage", decimalType(use_decimal, 5, 2))
    ])

    SCHEMAS["store_returns"] = StructType([
        StructField("sr_returned_date_sk", identifier_int),
        StructField("sr_return_time_sk", identifier_int),
        StructField("sr_item_sk", identifier_int, nullable=False),
        StructField("sr_customer_sk", identifier_int),
        StructField("sr_cdemo_sk", identifier_int),
        StructField("sr_hdemo_sk", identifier_int),
        StructField("sr_addr_sk", identifier_int),
        StructField("sr_store_sk", identifier_int),
        StructField("sr_reason_sk", identifier_int),
        # Use LongType due to https://github.com/NVIDIA/spark-rapids-benchmarks/pull/9#issuecomment-1138379596
        # Databricks is using LongType as well in their accepted benchmark reports.
        # See https://www.tpc.org/results/supporting_files/tpcds/databricks~tpcds~100000~databricks_SQL_8.3~sup-1~2021-11-02~v01.zip
        StructField("sr_ticket_number", identifier_long, nullable=False),
        StructField("sr_return_quantity", LongType()),
        StructField("sr_return_amt", decimalType(use_decimal, 7, 2)),
        StructField("sr_return_tax", decimalType(use_decimal, 7, 2)),
        StructField("sr_return_amt_inc_tax", decimalType(use_decimal, 7, 2)),
        StructField("sr_fee", decimalType(use_decimal, 7, 2)),
        StructField("sr_return_ship_cost", decimalType(use_decimal, 7, 2)),
        StructField("sr_refunded_cash", decimalType(use_decimal, 7, 2)),
        StructField("sr_reversed_charge", decimalType(use_decimal, 7, 2)),
        StructField("sr_store_credit", decimalType(use_decimal, 7, 2)),
        StructField("sr_net_loss", decimalType(use_decimal, 7, 2))
    ])

    SCHEMAS["household_demographics"] = StructType([
        StructField("hd_demo_sk", identifier_int, nullable=False),
        StructField("hd_income_band_sk", identifier_int),
        StructField("hd_buy_potential", CharType(15)),
        StructField("hd_dep_count", LongType()),
        StructField("hd_vehicle_count", LongType())
    ])

    SCHEMAS["web_page"] = StructType([
        StructField("wp_web_page_sk", identifier_int, nullable=False),
        StructField("wp_web_page_id", CharType(16), nullable=False),
        StructField("wp_rec_start_date", DateType()),
        StructField("wp_rec_end_date", DateType()),
        StructField("wp_creation_date_sk", identifier_int),
        StructField("wp_access_date_sk", identifier_int),
        StructField("wp_autogen_flag", CharType(1)),
        StructField("wp_customer_sk", identifier_int),
        StructField("wp_url", VarcharType(100)),
        StructField("wp_type", CharType(50)),
        StructField("wp_char_count", LongType()),
        StructField("wp_link_count", LongType()),
        StructField("wp_image_count", LongType()),
        StructField("wp_max_ad_count", LongType())
    ])

    SCHEMAS["promotion"] = StructType([
        StructField("p_promo_sk", identifier_int, nullable=False),
        StructField("p_promo_id", CharType(16), nullable=False),
        StructField("p_start_date_sk", identifier_int),
        StructField("p_end_date_sk", identifier_int),
        StructField("p_item_sk", identifier_int),
        StructField("p_cost", decimalType(use_decimal, 15, 2)),
        StructField("p_response_target", LongType()),
        StructField("p_promo_name", CharType(50)),
        StructField("p_channel_dmail", CharType(1)),
        StructField("p_channel_email", CharType(1)),
        StructField("p_channel_catalog", CharType(1)),
        StructField("p_channel_tv", CharType(1)),
        StructField("p_channel_radio", CharType(1)),
        StructField("p_channel_press", CharType(1)),
        StructField("p_channel_event", CharType(1)),
        StructField("p_channel_demo", CharType(1)),
        StructField("p_channel_details", VarcharType(100)),
        StructField("p_purpose", CharType(15)),
        StructField("p_discount_active", CharType(1))
    ])

    SCHEMAS["catalog_page"] = StructType([
        StructField("cp_catalog_page_sk", identifier_int, nullable=False),
        StructField("cp_catalog_page_id", CharType(16), nullable=False),
        StructField("cp_start_date_sk", identifier_int),
        StructField("cp_end_date_sk", identifier_int),
        StructField("cp_department", VarcharType(50)),
        StructField("cp_catalog_number", LongType()),
        StructField("cp_catalog_page_number", LongType()),
        StructField("cp_description", VarcharType(100)),
        StructField("cp_type", VarcharType(100))
    ])

    SCHEMAS["inventory"] = StructType([
        StructField("inv_date_sk", identifier_int, nullable=False),
        StructField("inv_item_sk", identifier_int, nullable=False),
        StructField("inv_warehouse_sk", identifier_int, nullable=False),
        StructField("inv_quantity_on_hand", LongType())
    ])

    SCHEMAS["catalog_returns"] = StructType([
        StructField("cr_returned_date_sk", identifier_int),
        StructField("cr_returned_time_sk", identifier_int),
        StructField("cr_item_sk", identifier_int, nullable=False),
        StructField("cr_refunded_customer_sk", identifier_int),
        StructField("cr_refunded_cdemo_sk", identifier_int),
        StructField("cr_refunded_hdemo_sk", identifier_int),
        StructField("cr_refunded_addr_sk", identifier_int),
        StructField("cr_returning_customer_sk", identifier_int),
        StructField("cr_returning_cdemo_sk", identifier_int),
        StructField("cr_returning_hdemo_sk", identifier_int),
        StructField("cr_returning_addr_sk", identifier_int),
        StructField("cr_call_center_sk", identifier_int),
        StructField("cr_catalog_page_sk", identifier_int),
        StructField("cr_ship_mode_sk", identifier_int),
        StructField("cr_warehouse_sk", identifier_int),
        StructField("cr_reason_sk", identifier_int),
        StructField("cr_order_number", identifier_int, nullable=False),
        StructField("cr_return_quantity", LongType()),
        StructField("cr_return_amount", decimalType(use_decimal, 7, 2)),
        StructField("cr_return_tax", decimalType(use_decimal, 7, 2)),
        StructField("cr_return_amt_inc_tax", decimalType(use_decimal, 7, 2)),
        StructField("cr_fee", decimalType(use_decimal, 7, 2)),
        StructField("cr_return_ship_cost", decimalType(use_decimal, 7, 2)),
        StructField("cr_refunded_cash", decimalType(use_decimal, 7, 2)),
        StructField("cr_reversed_charge", decimalType(use_decimal, 7, 2)),
        StructField("cr_store_credit", decimalType(use_decimal, 7, 2)),
        StructField("cr_net_loss", decimalType(use_decimal, 7, 2))
    ])

    SCHEMAS["web_returns"] = StructType([
        StructField("wr_returned_date_sk", identifier_int),
        StructField("wr_returned_time_sk", identifier_int),
        StructField("wr_item_sk", identifier_int, nullable=False),
        StructField("wr_refunded_customer_sk", identifier_int),
        StructField("wr_refunded_cdemo_sk", identifier_int),
        StructField("wr_refunded_hdemo_sk", identifier_int),
        StructField("wr_refunded_addr_sk", identifier_int),
        StructField("wr_returning_customer_sk", identifier_int),
        StructField("wr_returning_cdemo_sk", identifier_int),
        StructField("wr_returning_hdemo_sk", identifier_int),
        StructField("wr_returning_addr_sk", identifier_int),
        StructField("wr_web_page_sk", identifier_int),
        StructField("wr_reason_sk", identifier_int),
        StructField("wr_order_number", identifier_int, nullable=False),
        StructField("wr_return_quantity", LongType()),
        StructField("wr_return_amt", decimalType(use_decimal, 7, 2)),
        StructField("wr_return_tax", decimalType(use_decimal, 7, 2)),
        StructField("wr_return_amt_inc_tax", decimalType(use_decimal, 7, 2)),
        StructField("wr_fee", decimalType(use_decimal, 7, 2)),
        StructField("wr_return_ship_cost", decimalType(use_decimal, 7, 2)),
        StructField("wr_refunded_cash", decimalType(use_decimal, 7, 2)),
        StructField("wr_reversed_charge", decimalType(use_decimal, 7, 2)),
        StructField("wr_account_credit", decimalType(use_decimal, 7, 2)),
        StructField("wr_net_loss", decimalType(use_decimal, 7, 2))
    ])

    SCHEMAS["web_sales"] = StructType([
        StructField("ws_sold_date_sk", identifier_int),
        StructField("ws_sold_time_sk", identifier_int),
        StructField("ws_ship_date_sk", identifier_int),
        StructField("ws_item_sk", identifier_int, nullable=False),
        StructField("ws_bill_customer_sk", identifier_int),
        StructField("ws_bill_cdemo_sk", identifier_int),
        StructField("ws_bill_hdemo_sk", identifier_int),
        StructField("ws_bill_addr_sk", identifier_int),
        StructField("ws_ship_customer_sk", identifier_int),
        StructField("ws_ship_cdemo_sk", identifier_int),
        StructField("ws_ship_hdemo_sk", identifier_int),
        StructField("ws_ship_addr_sk", identifier_int),
        StructField("ws_web_page_sk", identifier_int),
        StructField("ws_web_site_sk", identifier_int),
        StructField("ws_ship_mode_sk", identifier_int),
        StructField("ws_warehouse_sk", identifier_int),
        StructField("ws_promo_sk", identifier_int),
        StructField("ws_order_number", identifier_int, nullable=False),
        StructField("ws_quantity", LongType()),
        StructField("ws_wholesale_cost", decimalType(use_decimal, 7, 2)),
        StructField("ws_list_price", decimalType(use_decimal, 7, 2)),
        StructField("ws_sales_price", decimalType(use_decimal, 7, 2)),
        StructField("ws_ext_discount_amt", decimalType(use_decimal, 7, 2)),
        StructField("ws_ext_sales_price", decimalType(use_decimal, 7, 2)),
        StructField("ws_ext_wholesale_cost", decimalType(use_decimal, 7, 2)),
        StructField("ws_ext_list_price", decimalType(use_decimal, 7, 2)),
        StructField("ws_ext_tax", decimalType(use_decimal, 7, 2)),
        StructField("ws_coupon_amt", decimalType(use_decimal, 7, 2)),
        StructField("ws_ext_ship_cost", decimalType(use_decimal, 7, 2)),
        StructField("ws_net_paid", decimalType(use_decimal, 7, 2)),
        StructField("ws_net_paid_inc_tax", decimalType(use_decimal, 7, 2)),
        StructField("ws_net_paid_inc_ship", decimalType(use_decimal, 7, 2)),
        StructField("ws_net_paid_inc_ship_tax",
                    decimalType(use_decimal, 7, 2)),
        StructField("ws_net_profit", decimalType(use_decimal, 7, 2))
    ])

    SCHEMAS["catalog_sales"] = StructType([
        StructField("cs_sold_date_sk", identifier_int),
        StructField("cs_sold_time_sk", identifier_int),
        StructField("cs_ship_date_sk", identifier_int),
        StructField("cs_bill_customer_sk", identifier_int),
        StructField("cs_bill_cdemo_sk", identifier_int),
        StructField("cs_bill_hdemo_sk", identifier_int),
        StructField("cs_bill_addr_sk", identifier_int),
        StructField("cs_ship_customer_sk", identifier_int),
        StructField("cs_ship_cdemo_sk", identifier_int),
        StructField("cs_ship_hdemo_sk", identifier_int),
        StructField("cs_ship_addr_sk", identifier_int),
        StructField("cs_call_center_sk", identifier_int),
        StructField("cs_catalog_page_sk", identifier_int),
        StructField("cs_ship_mode_sk", identifier_int),
        StructField("cs_warehouse_sk", identifier_int),
        StructField("cs_item_sk", identifier_int, nullable=False),
        StructField("cs_promo_sk", identifier_int),
        StructField("cs_order_number", identifier_int, nullable=False),
        StructField("cs_quantity", LongType()),
        StructField("cs_wholesale_cost", decimalType(use_decimal, 7, 2)),
        StructField("cs_list_price", decimalType(use_decimal, 7, 2)),
        StructField("cs_sales_price", decimalType(use_decimal, 7, 2)),
        StructField("cs_ext_discount_amt", decimalType(use_decimal, 7, 2)),
        StructField("cs_ext_sales_price", decimalType(use_decimal, 7, 2)),
        StructField("cs_ext_wholesale_cost", decimalType(use_decimal, 7, 2)),
        StructField("cs_ext_list_price", decimalType(use_decimal, 7, 2)),
        StructField("cs_ext_tax", decimalType(use_decimal, 7, 2)),
        StructField("cs_coupon_amt", decimalType(use_decimal, 7, 2)),
        StructField("cs_ext_ship_cost", decimalType(use_decimal, 7, 2)),
        StructField("cs_net_paid", decimalType(use_decimal, 7, 2)),
        StructField("cs_net_paid_inc_tax", decimalType(use_decimal, 7, 2)),
        StructField("cs_net_paid_inc_ship", decimalType(use_decimal, 7, 2)),
        StructField("cs_net_paid_inc_ship_tax",
                    decimalType(use_decimal, 7, 2)),
        StructField("cs_net_profit", decimalType(use_decimal, 7, 2))
    ])

    SCHEMAS["store_sales"] = StructType([
        StructField("ss_sold_date_sk", identifier_int),
        StructField("ss_sold_time_sk", identifier_int),
        StructField("ss_item_sk", identifier_int, nullable=False),
        StructField("ss_customer_sk", identifier_int),
        StructField("ss_cdemo_sk", identifier_int),
        StructField("ss_hdemo_sk", identifier_int),
        StructField("ss_addr_sk", identifier_int),
        StructField("ss_store_sk", identifier_int),
        StructField("ss_promo_sk", identifier_int),
        StructField("ss_ticket_number", identifier_long, nullable=False),
        StructField("ss_quantity", LongType()),
        StructField("ss_wholesale_cost", decimalType(use_decimal, 7, 2)),
        StructField("ss_list_price", decimalType(use_decimal, 7, 2)),
        StructField("ss_sales_price", decimalType(use_decimal, 7, 2)),
        StructField("ss_ext_discount_amt", decimalType(use_decimal, 7, 2)),
        StructField("ss_ext_sales_price", decimalType(use_decimal, 7, 2)),
        StructField("ss_ext_wholesale_cost", decimalType(use_decimal, 7, 2)),
        StructField("ss_ext_list_price", decimalType(use_decimal, 7, 2)),
        StructField("ss_ext_tax", decimalType(use_decimal, 7, 2)),
        StructField("ss_coupon_amt", decimalType(use_decimal, 7, 2)),
        StructField("ss_net_paid", decimalType(use_decimal, 7, 2)),
        StructField("ss_net_paid_inc_tax", decimalType(use_decimal, 7, 2)),
        StructField("ss_net_profit", decimalType(use_decimal, 7, 2))
    ])
    return SCHEMAS

def get_maintenance_schemas(use_decimal):
    MAINTENANCE_SCHEMAS = {}
    MAINTENANCE_SCHEMAS["s_purchase_lineitem"] = StructType([
        StructField("plin_purchase_id", IntegerType(), nullable=False),
        StructField("plin_line_number", IntegerType(), nullable=False),
        StructField("plin_item_id", CharType(16)),
        StructField("plin_promotion_id", CharType(16)),
        StructField("plin_quantity", IntegerType()),
        StructField("plin_sale_price", decimalType(use_decimal, 7,2)),
        StructField("plin_coupon_amt", decimalType(use_decimal, 7,2)),
        StructField("plin_comment", VarcharType(100)),
    ])
    MAINTENANCE_SCHEMAS["s_purchase"] = StructType([
        StructField("purc_purchase_id", IntegerType(), nullable=False),
        StructField("purc_store_id", CharType(16)),
        StructField("purc_customer_id", CharType(16)),
        StructField("purc_purchase_date", CharType(10)),
        StructField("purc_purchase_time", IntegerType()),
        StructField("purc_register_id", IntegerType()),
        StructField("purc_clerk_id", IntegerType()),
        StructField("purc_comment", CharType(100)),
    ])
    MAINTENANCE_SCHEMAS["s_catalog_order"] = StructType([
        StructField("cord_order_id", IntegerType(), nullable=False),
        StructField("cord_bill_customer_id", CharType(16)),
        StructField("cord_ship_customer_id", CharType(16)),
        StructField("cord_order_date", CharType(10)),
        StructField("cord_order_time", IntegerType()),
        StructField("cord_ship_mode_id", CharType(16)),
        StructField("cord_call_center_id", CharType(16)),
        StructField("cord_order_comments", VarcharType(100)),
    ])
    MAINTENANCE_SCHEMAS["s_web_order"] = StructType([
        StructField("word_order_id", IntegerType(), nullable=False),
        StructField("word_bill_customer_id", CharType(16)),
        StructField("word_ship_customer_id", CharType(16)),
        StructField("word_order_date", CharType(10)),
        StructField("word_order_time", IntegerType()),
        StructField("word_ship_mode_id", CharType(16)),
        StructField("word_web_site_id", CharType(16)),
        StructField("word_order_comments", CharType(100)),
    ])
    MAINTENANCE_SCHEMAS["s_catalog_order_lineitem"] = StructType([
        StructField("clin_order_id", IntegerType(), nullable=False),
        StructField("clin_line_number", IntegerType(), nullable=False),
        StructField("clin_item_id", CharType(16)),
        StructField("clin_promotion_id", CharType(16)),
        StructField("clin_quantity", IntegerType()),
        StructField("clin_sales_price", decimalType(use_decimal, 7,2)),
        StructField("clin_coupon_amt", decimalType(use_decimal, 7,2)),
        StructField("clin_warehouse_id", CharType(16)),
        StructField("clin_ship_date", CharType(10)),
        StructField("clin_catalog_number", IntegerType()),
        StructField("clin_catalog_page_number", IntegerType()),
        StructField("clin_ship_cost", decimalType(use_decimal, 7,2)),
    ])
    MAINTENANCE_SCHEMAS["s_web_order_lineitem"] = StructType([
        StructField("wlin_order_id", IntegerType(), nullable=False),
        StructField("wlin_line_number", IntegerType(), nullable=False),
        StructField("wlin_item_id", CharType(16)),
        StructField("wlin_promotion_id", CharType(16)),
        StructField("wlin_quantity", IntegerType()),
        StructField("wlin_sales_price", decimalType(use_decimal, 7,2)),
        StructField("wlin_coupon_amt", decimalType(use_decimal, 7,2)),
        StructField("wlin_warehouse_id", CharType(16)),
        StructField("wlin_ship_date", CharType(10)),
        StructField("wlin_ship_cost", decimalType(use_decimal, 7,2)),
        StructField("wlin_web_page_id", CharType(16)),
    ])
    MAINTENANCE_SCHEMAS["s_store_returns"] = StructType([
        StructField("sret_store_id", CharType(16)),
        StructField("sret_purchase_id", CharType(16), nullable=False),
        StructField("sret_line_number", IntegerType(), nullable=False),
        StructField("sret_item_id", CharType(16), nullable=False),
        StructField("sret_customer_id", CharType(16)),
        StructField("sret_return_date", CharType(10)),
        StructField("sret_return_time", CharType(10)),
        StructField("sret_ticket_number", LongType()),
        StructField("sret_return_qty", IntegerType()),
        StructField("sret_return_amt", decimalType(use_decimal, 7,2)),
        StructField("sret_return_tax", decimalType(use_decimal, 7,2)),
        StructField("sret_return_fee", decimalType(use_decimal, 7,2)),
        StructField("sret_return_ship_cost", decimalType(use_decimal, 7,2)),
        StructField("sret_refunded_cash", decimalType(use_decimal, 7,2)),
        StructField("sret_reversed_charge", decimalType(use_decimal, 7,2)),
        StructField("sret_store_credit", decimalType(use_decimal, 7,2)),
        StructField("sret_reason_id", CharType(16)),
    ])
    MAINTENANCE_SCHEMAS["s_catalog_returns"] = StructType([
        StructField("cret_call_center_id", CharType(16)),
        StructField("cret_order_id", IntegerType(), nullable=False),
        StructField("cret_line_number", IntegerType(), nullable=False),
        StructField("cret_item_id", CharType(16), nullable=False),
        StructField("cret_return_customer_id", CharType(16)),
        StructField("cret_refund_customer_id", CharType(16)),
        StructField("cret_return_date", CharType(10)),
        StructField("cret_return_time", CharType(10)),
        StructField("cret_return_qty", IntegerType()),
        StructField("cret_return_amt", decimalType(use_decimal, 7,2)),
        StructField("cret_return_tax", decimalType(use_decimal, 7,2)),
        StructField("cret_return_fee", decimalType(use_decimal, 7,2)),
        StructField("cret_return_ship_cost", decimalType(use_decimal, 7,2)),
        StructField("cret_refunded_cash", decimalType(use_decimal, 7,2)),
        StructField("cret_reversed_charge", decimalType(use_decimal, 7,2)),
        StructField("cret_merchant_credit", decimalType(use_decimal, 7,2)),
        StructField("cret_reason_id", CharType(16)),
        StructField("cret_shipmode_id", CharType(16)),
        StructField("cret_catalog_page_id", CharType(16)),
        StructField("cret_warehouse_id", CharType(16)),
    ])
    MAINTENANCE_SCHEMAS["s_web_returns"] = StructType([
         StructField("wret_web_page_id", CharType(16)),
         StructField("wret_order_id", IntegerType(), nullable=False),
         StructField("wret_line_number", IntegerType(), nullable=False),
         StructField("wret_item_id", CharType(16), nullable=False),
         StructField("wret_return_customer_id", CharType(16)),
         StructField("wret_refund_customer_id", CharType(16)),
         StructField("wret_return_date", CharType(10)),
         StructField("wret_return_time", CharType(10)),
         StructField("wret_return_qty", IntegerType()),
         StructField("wret_return_amt", decimalType(use_decimal,7,2)),
         StructField("wret_return_tax", decimalType(use_decimal,7,2)),
         StructField("wret_return_fee", decimalType(use_decimal,7,2)),
         StructField("wret_return_ship_cost", decimalType(use_decimal,7,2)),
         StructField("wret_refunded_cash", decimalType(use_decimal,7,2)),
         StructField("wret_reversed_charge", decimalType(use_decimal,7,2)),
         StructField("wret_account_credit", decimalType(use_decimal,7,2)),
         StructField("wret_reason_id", CharType(16)),
    ])

    MAINTENANCE_SCHEMAS["s_inventory"] = StructType([
        StructField("invn_warehouse_id", CharType(16), nullable=False),
        StructField("invn_item_id", CharType(16), nullable=False),
        StructField("invn_date", CharType(10), nullable=False),
        StructField("invn_qty_on_hand", IntegerType()),
    ])

    MAINTENANCE_SCHEMAS["delete"] = StructType([
        StructField("date1", StringType(), nullable=False),
        StructField("date2", StringType(), nullable=False),
    ])

    MAINTENANCE_SCHEMAS["inventory_delete"] = StructType([
        StructField("date1", StringType(), nullable=False),
        StructField("date2", StringType(), nullable=False),
    ])
    return MAINTENANCE_SCHEMAS

if __name__ == "__main__":
    # Test code
    print(get_schemas(False))
    print(get_schemas(True))
    print(get_maintenance_schemas(False))
    print(get_maintenance_schemas(True))