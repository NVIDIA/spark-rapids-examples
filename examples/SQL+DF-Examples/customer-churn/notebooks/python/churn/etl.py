#!/usr/bin/env python
# coding: utf-8

# Copyright (c) 2022, NVIDIA Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pyspark
import pyspark.sql
import pyspark.sql.functions as F

from collections import defaultdict

options = defaultdict(lambda: None)
session = None

ETL_VERSION = '0.7'

def register_options(**kwargs):
    global options
    for k, v in kwargs.items():
        options[k] = v

def _register_session(s):
    global session
    session = s

def _register_views(lvars, *names):
    for n in names:
        if n in lvars:
            lvars[n].createOrReplaceTempView(n)

def withsession(df_arg=0):
    def decorate(fn):
        def wrapped(*args, **kwargs):
            _register_session(args[df_arg].sql_ctx.sparkSession)
            fn(*args, **kwargs)
        return wrapped
    return decorate

def read_df(session, fn):
    kwargs = {}
    _register_session(session)
    input_kind = options["input_kind"]

    if input_kind == "csv":
        kwargs["header"] = True
    return getattr(session.read, input_kind)("%s.%s" % (fn, input_kind), **kwargs)


def find_customers(billing_events_df):
    customers = billing_events_df.select("customerID").distinct()
    if 'cache_customers' in options:
        customers.cache()
    customers.createOrReplaceTempView("customers")
    return customers

def customers():
    global session
    return session.table("customers")

def join_billing_data(billing_events_df):
    _register_session(billing_events_df.sql_ctx.sparkSession)

    billing_events = billing_events_df.withColumn("value", billing_events_df.value)

    customers = find_customers(billing_events)

    counts_and_charges = billing_events.groupBy("customerID", "kind").agg(
        F.count(billing_events.value).alias("event_counts"),
        F.sum(billing_events.value).alias("total_charges"),
    )

    counts_and_charges.createOrReplaceTempView("counts_and_charges")
    
    terminations = billing_events.where(F.col("kind") == "AccountTermination").select(
        F.col("customerID").alias("Churn")
    )

    churned = customers.join(
        terminations, customers.customerID == terminations.Churn, how="leftouter"
    ).select(
        "customerID", F.when(F.col("Churn").isNull(), F.lit(False)).otherwise(F.lit(True)).alias("Churn")
    )

    customer_charges = customers.join(
        counts_and_charges.where(F.col("kind") == "Charge"), "customerID", how="leftouter"
    ).select(
        "customerID",
        F.col("event_counts").alias("tenure"),
        F.col("total_charges").alias("TotalCharges"),
    ).fillna({'tenure': 0, 'TotalCharges': 0.0})

    _register_views(locals(), "counts_and_charges", "terminations", "churned", "customer_charges")
 
    # counts_and_charges.createOrReplaceTempView("counts_and_charges")
    # terminations.createOrReplaceTempView("terminations")
    # churned.createOrReplaceTempView("churned")
    # customer_charges.createOrReplaceTempView("customer_charges")
    
    customer_billing = churned.join(customer_charges, "customerID")
    _register_views(locals(), "counts_and_charges", "terminations", "churned", "customer_charges", "customer_billing")
 
    return customer_billing


def join_phone_features(phone_features_df):
    phone_features = phone_features_df

    phone_service = phone_features.where(F.col("feature") == "PhoneService").select(
        "customerID", F.lit("Yes").alias("PhoneService")
    )

    multiple_lines = phone_features.where(F.col("feature") == "MultipleLines").select(
        "customerID", F.lit("Yes").alias("MultipleLines")
    )

    customer_phone_features = (
        customers().join(phone_service, "customerID", how="leftouter")
        .join(multiple_lines, "customerID", how="leftouter")
        .select(
            "customerID",
            F.when(F.col("PhoneService").isNull(), "No")
            .otherwise("Yes")
            .alias("PhoneService"),
            "MultipleLines",
        )
        .select(
            "customerID",
            "PhoneService",
            F.when(F.col("PhoneService") == "No", "No phone service")
            .otherwise(F.when(F.col("MultipleLines").isNull(), "No").otherwise("Yes"))
            .alias("MultipleLines"),
        )
    )

    _register_views(locals(), "phone_service", "multiple_lines", "customer_phone_features")
 
    return customer_phone_features


def untidy_feature(df, feature):
    """ 'untidies' a feature by turning it into a column """
    return df.where(F.col("feature") == feature).select(
        "customerID", F.col("value").alias(feature)
    )

def chained_join(column, base_df, dfs, how="leftouter"):
    """ repeatedly joins a sequence of data frames on the same column """
    acc = base_df
    for df in dfs:
        acc = acc.join(df, column, how=how)

    return acc

def resolve_nullable_column(df, col, null_val="No"):
    return F.when(df[col].isNull(), null_val).otherwise(df[col]).alias(col)


def resolve_dependent_column(
    df,
    col,
    parent_col="InternetService",
    null_val="No",
    null_parent_val="No internet service",
):
    return (
        F.when((df[parent_col] == "No") | (df[parent_col].isNull()), null_parent_val)
        .otherwise(F.when(df[col].isNull(), null_val).otherwise(df[col]))
        .alias(col)
    )


def join_internet_features(internet_features_df):

    internet_features = internet_features_df

    internet_service = untidy_feature(internet_features, "InternetService")
    online_security = untidy_feature(internet_features, "OnlineSecurity")
    online_backup = untidy_feature(internet_features, "OnlineBackup")
    device_protection = untidy_feature(internet_features, "DeviceProtection")
    tech_support = untidy_feature(internet_features, "TechSupport")
    streaming_tv = untidy_feature(internet_features, "StreamingTV")
    streaming_movies = untidy_feature(internet_features, "StreamingMovies")

    customer_internet_features = chained_join(
        "customerID",
        customers(),
        [
            internet_service,
            online_security,
            online_backup,
            device_protection,
            tech_support,
            streaming_tv,
            streaming_movies,
        ],
    )
    
    customer_internet_features = customer_internet_features.select(
        "customerID",
        resolve_nullable_column(customer_internet_features, "InternetService"),
        resolve_dependent_column(
            customer_internet_features, "OnlineSecurity", "InternetService"
        ),
        resolve_dependent_column(
            customer_internet_features, "OnlineBackup", "InternetService"
        ),
        resolve_dependent_column(
            customer_internet_features, "DeviceProtection", "InternetService"
        ),
        resolve_dependent_column(
            customer_internet_features, "TechSupport", "InternetService"
        ),
        resolve_dependent_column(
            customer_internet_features, "StreamingTV", "InternetService"
        ),
        resolve_dependent_column(
            customer_internet_features, "StreamingMovies", "InternetService"
        ),
    )

    _register_views(locals(), 
        "internet_service",
        "online_security",
        "online_backup",
        "device_protection",
        "tech_support",
        "streaming_tv",
        "streaming_movies",
        "customer_internet_features" 
    )

    return customer_internet_features


def join_account_features(account_features_df):
    account_features = account_features_df
    contracts = untidy_feature(account_features, "Contract")

    paperless = untidy_feature(account_features, "PaperlessBilling")

    payment = untidy_feature(account_features, "PaymentMethod")

    customer_account_features = chained_join(
        "customerID", customers(), [contracts, paperless, payment]
    )

    customer_account_features = customer_account_features.select(
        "customerID",
        "Contract",
        resolve_nullable_column(customer_account_features, "PaperlessBilling"),
        "PaymentMethod",
    )

    _register_views(locals(), "contracts", "paperless", "payment", "customer_account_features")
    
    return customer_account_features


def process_account_meta(account_meta_df, usecal=None):
    def is_senior_citizen(nowcol, dobcol):
        if options['use_calendar_arithmetic']:
            return F.when(
                F.col("now") >= F.add_months(
                    F.col("dateOfBirth"), 65 * 12
                ), F.lit(True)
            ).otherwise(F.lit(False))
        else:
            return (F.year(F.col(nowcol)) > (F.year(F.col(dobcol)) + 65)) | \
                (F.year(F.col(nowcol)) == (F.year(F.col(dobcol)) + 65)) & \
                (
                    (F.month(F.col(nowcol)) < F.month(F.col(dobcol))) | \
                    (
                        (F.month(F.col(nowcol)) == F.month(F.col(dobcol))) & \
                        (F.dayofmonth(F.col(nowcol)) <= F.dayofmonth(F.col(nowcol)))
                    )
                )

    customer_account_meta = account_meta_df.select(
        "customerID",
        is_senior_citizen("now", "dateOfBirth").alias("SeniorCitizen"),
        "Partner",
        "Dependents",
        "gender",
        "MonthlyCharges",
    )
    
    _register_views(locals(), "customer_account_meta")
    return customer_account_meta

def forcefloat(c):
    return F.col(c).cast("float").alias(c)


def join_wide_table(customer_billing, customer_phone_features, customer_internet_features, customer_account_features, customer_account_meta):

    wide_data = chained_join(
        "customerID",
        customers(),
        [
            customer_billing,
            customer_phone_features,
            customer_internet_features,
            customer_account_features,
            customer_account_meta,
        ],
    ).select(
        "customerID",
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "tenure",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
        "MonthlyCharges",
        "TotalCharges",
        "Churn",
    )

    return wide_data

    # In[ ]:

def cast_and_coalesce_wide_data(wd):
    if options["coalesce_output"] > 0:
        wd = wd.coalesce(options["coalesce_output"])
    
    return wd.select(
        "customerID",
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "tenure",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
        forcefloat("MonthlyCharges"),
        forcefloat("TotalCharges"),
        "Churn",
    )

def write_df(df, name):
    output_kind = options["output_kind"]
    output_mode = options["output_mode"]
    output_prefix = options["output_prefix"]
    
    name = "%s.%s" % (name, output_kind)
    if output_prefix != "":
        name = "%s%s" % (output_prefix, name)
    kwargs = {}
    if output_kind == "csv":
        kwargs["header"] = True
    getattr(df.write.mode(output_mode), output_kind)(name, **kwargs)

