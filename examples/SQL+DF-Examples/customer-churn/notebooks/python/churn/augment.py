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

import datetime
import os

import pyspark
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, DecimalType
import pyspark.sql.functions as F
from collections import defaultdict

options = defaultdict(lambda: None)

now = datetime.datetime.now(datetime.timezone.utc)

AUGMENT_VERSION = "0.7"
AUGMENT_CUSTOMER_TAG = "0007"

session = None
currencyType = None

def get_currency_type():
    global options
    global currencyType

    if currencyType is not None:
        return currencyType
    
    if "use_decimal" in options and options["use_decimal"]:
        if "decimal_precision" in options :
            assert options["decimal_precision"] > 5, "Decimal precision is too small; was %d but should be at least 6" % options["decimal_precision"]
            currencyType = DecimalType(options["decimal_precision"], 2)
        else:
            # "999,999.99 should be enough for anyone"
            currencyType = DecimalType(8, 2)
    else:
        currencyType = DoubleType()
    
    return currencyType

def _register_session(s):
    global session
    session = s

def _get_uniques(ct):
    global session
    table_names = set([table.name for table in session.catalog.listTables()])

    if ("uniques_%d" % ct) in table_names:
        return session.table("uniques_%d" % ct)
    else:
        def str_part(seed=0x5CA1AB1E):
            "generate the string part of a unique ID"
            import random

            r = random.Random(seed)
            from base64 import b64encode

            while True:
                yield "%s-%s" % (b64encode(r.getrandbits(72).to_bytes(9, "big"), b"@_").decode(
                    "utf-8"
                ), AUGMENT_CUSTOMER_TAG)
        
        sp = str_part()
        
        uniques = (
            session.createDataFrame(
                schema=StructType([StructField("u_value", StringType())]),
                data=[dict(u_value=next(sp)) for _ in range(min(int(ct * 1.02), ct + 2))],
            )
            .distinct()
            .orderBy("u_value")
            .limit(ct)
        ).cache()

        uc = uniques.count()
        assert (uc == ct), "due to prng collision we had %d instead of %d replicas" % (uc, ct)

        uniques.createOrReplaceTempView("uniques_%d" % ct)

        return uniques
        

def register_options(**kwargs):
    global options
    for k, v in kwargs.items():
        options[k] = v

def load_supplied_data(session, input_file):
    _register_session(session)

    fields = [
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
    ]
    double_fields = set(["tenure", "MonthlyCharges", "TotalCharges"])

    schema = pyspark.sql.types.StructType(
        [
            pyspark.sql.types.StructField(
                f, DoubleType() if f in double_fields else StringType()
            )
            for f in fields
        ]
    )

    df = session.read.csv(input_file, header=True, schema=schema)
    
    source_count = df.count()
    df = df.dropna()
    nn_count = df.count()

    if source_count == nn_count:    
        print("read %d records from source dataset with no nulls -- is this what you expect?" % source_count)
    else:
        print("read %d records from source dataset (%d non-null records)" % (source_count, nn_count))
    
    return df

def replicate_df(df, duplicates):

    if duplicates > 1:
        uniques = _get_uniques(duplicates)

        df = (
            df.crossJoin(uniques.distinct())
            .withColumn("customerID", F.format_string("%s-%s", "customerID", "u_value"))
            .drop("u_value")
        )

    return df

def examine_categoricals(df, columns=None):
    """ Returns (to driver memory) a list of tuples consisting of every unique value 
        for each column in `columns` or for every categorical column in the source 
        data if no columns are specified """
    default_columns = [
        "SeniorCitizen",
        "Partner",
        "Dependents",
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
    ]

    columns = columns or default_columns

    return [(c, [row[0] for row in df.select(c).distinct().rdd.collect()]) for c in columns]

def billing_events(df):
    import datetime

    MAX_MONTH = 72

    def get_last_month(col):
        h = F.abs(F.xxhash64(col))
        h1 = (h.bitwiseAND(0xff)) % (MAX_MONTH // 2)
        h2 = (F.shiftRight(h, 8).bitwiseAND(0xff)) % (MAX_MONTH // 3)
        h3 = (F.shiftRight(h, 16).bitwiseAND(0xff)) % (MAX_MONTH // 5)
        h4 = (F.shiftRight(h, 24).bitwiseAND(0xff)) % (MAX_MONTH // 7)
        h5 = (F.shiftRight(h, 32).bitwiseAND(0xff)) % (MAX_MONTH // 11)
        return -(h1 + h2 + h3 + h4 + h5)

    w = pyspark.sql.Window.orderBy(F.lit("")).partitionBy(df.customerID)

    charges = (
        df.select(
            df.customerID,
            F.lit("Charge").alias("kind"),
            F.explode(
                F.array_repeat((df.TotalCharges / df.tenure).cast(get_currency_type()), df.tenure.cast("int"))
            ).alias("value"),
            F.when(df.Churn == "Yes", get_last_month(df.customerID)).otherwise(0).alias("last_month")
        )
        .withColumn("now", F.lit(now).cast("date"))
        .withColumn("month_number", -(F.row_number().over(w) + F.col("last_month")))
        .withColumn("date", F.expr("add_months(now, month_number)"))
        .drop("now", "month_number", "last_month")
    )

    serviceStarts = (
        df.withColumn("last_month", F.when(df.Churn == "Yes", get_last_month(df.customerID)).otherwise(0)).select(
            df.customerID,
            F.lit("AccountCreation").alias("kind"),
            F.lit(0.0).cast(get_currency_type()).alias("value"),
            F.lit(now).alias("now"),
            (-df.tenure - 1 + F.col("last_month")).alias("month_number"),
        )
        .withColumn("date", F.expr("add_months(now, month_number)"))
        .drop("now", "month_number")
    )

    serviceTerminations = df.withColumn("last_month", F.when(df.Churn == "Yes", get_last_month(df.customerID)).otherwise(0)).where(
        df.Churn == "Yes"
    ).withColumn("now", F.lit(now)).select(
        df.customerID,
        F.lit("AccountTermination").alias("kind"),
        F.lit(0.0).cast(get_currency_type()).alias("value"),
        F.expr("add_months(now, last_month)").alias("date")
    )

    billingEvents = charges.union(serviceStarts).union(serviceTerminations).orderBy("date").withColumn("month", F.substring("date", 0, 7))
    return billingEvents

def resolve_path(name):
    output_prefix = options["output_prefix"] or ""
    output_mode = options["output_mode"] or "overwrite"
    output_kind = options["output_kind"] or "parquet"
    name = "%s.%s" % (name, output_kind)
    if output_prefix != "":
        name = "%s%s" % (output_prefix, name)
    
    return name

def write_df(df, name, skip_replication=False, partition_by=None):
    dup_times = options["dup_times"] or 1
    output_prefix = options["output_prefix"] or ""
    output_mode = options["output_mode"] or "overwrite"
    output_kind = options["output_kind"] or "parquet"

    if not skip_replication:
        df = replicate_df(df, dup_times)
    write = df.write
    if partition_by is not None:
        if type(partition_by) == str:
            partition_by = [partition_by]
        write = write.partitionBy(*partition_by)
    name = "%s.%s" % (name, output_kind)
    if output_prefix != "":
        name = "%s%s" % (output_prefix, name)
    kwargs = {}
    if output_kind == "csv":
        kwargs["header"] = True
    getattr(write.mode(output_mode), output_kind)(name, **kwargs)

def customer_meta(df):
    SENIOR_CUTOFF = 65
    ADULT_CUTOFF = 18
    DAYS_IN_YEAR = 365.25
    EXPONENTIAL_DIST_SCALE = 6.3

    augmented_original = replicate_df(df, options["dup_times"] or 1)

    customerMetaRaw = augmented_original.select(
        "customerID",
        F.lit(now).alias("now"),
        (F.abs(F.hash(augmented_original.customerID)) % 4096 / 4096).alias("choice"),
        "SeniorCitizen",
        "gender",
        "Partner",
        "Dependents",
        F.col("MonthlyCharges").cast(get_currency_type()).alias("MonthlyCharges"),
    )

    customerMetaRaw = customerMetaRaw.withColumn(
        "ageInDays",
        F.floor(
            F.when(
                customerMetaRaw.SeniorCitizen == 0,
                (
                    customerMetaRaw.choice
                    * ((SENIOR_CUTOFF - ADULT_CUTOFF - 1) * DAYS_IN_YEAR)
                )
                + (ADULT_CUTOFF * DAYS_IN_YEAR),
            ).otherwise(
                (SENIOR_CUTOFF * DAYS_IN_YEAR)
                + (
                    DAYS_IN_YEAR
                    * (-F.log1p(-customerMetaRaw.choice) * EXPONENTIAL_DIST_SCALE)
                )
            )
        ).cast("int"),
    )

    customerMetaRaw = customerMetaRaw.withColumn(
        "dateOfBirth", F.expr("date_sub(now, ageInDays)")
    )

    return customerMetaRaw.select(
        "customerID",
        "dateOfBirth",
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "MonthlyCharges",
        "now",
    ).orderBy("customerID")


def phone_features(df):
    phoneService = df.select(
        "customerID", F.lit("PhoneService").alias("feature"), F.lit("Yes").alias("value")
    ).where(df.PhoneService == "Yes")

    multipleLines = df.select(
        "customerID", F.lit("MultipleLines").alias("feature"), F.lit("Yes").alias("value")
    ).where(df.MultipleLines == "Yes")

    return phoneService.union(multipleLines).orderBy("customerID")

def internet_features(df):
    internet_service = df.select(
        "customerID",
        F.lit("InternetService").alias("feature"),
        df.InternetService.alias("value"),
    ).where(df.InternetService != "No")

    customerInternetFeatures = internet_service

    for feature in [
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]:
        tmpdf = df.select(
            "customerID",
            F.lit(feature).alias("feature"),
            df[feature].alias("value"),
        ).where(df[feature] == "Yes")

        customerInternetFeatures = customerInternetFeatures.union(tmpdf)

    return customerInternetFeatures


def account_features(df):
    session = df.sql_ctx.sparkSession
    accountSchema = pyspark.sql.types.StructType(
        [
            pyspark.sql.types.StructField(f, StringType())
            for f in ["customerID", "feature", "value"]
        ]
    )

    customerAccountFeatures = session.createDataFrame(schema=accountSchema, data=[])

    for feature in ["Contract", "PaperlessBilling", "PaymentMethod"]:
        tmpdf = df.select(
            "customerID",
            F.lit(feature).alias("feature"),
            df[feature].alias("value"),
        ).where(df[feature] != "No")

        customerAccountFeatures = customerAccountFeatures.union(tmpdf)
    
    return customerAccountFeatures


def debug_augmentation(df):
    return (
        df.select("customerID")
        .distinct()
        .select(
            "customerID",
            F.substring("customerID", 0, 10).alias("originalID"),
            F.element_at(F.split("customerID", "-", -1), 3).alias("suffix"),
        )
    )