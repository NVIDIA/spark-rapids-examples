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

from pyspark.sql import types as T
from pyspark.sql import functions as F

eda_options = { 'use_array_ops' : False }

def isnumeric(data_type):
    numeric_types = [T.ByteType, T.ShortType, T.IntegerType, T.LongType, T.FloatType, T.DoubleType, T.DecimalType]
    return any([isinstance(data_type, t) for t in numeric_types])


def percent_true(df, cols):
    denominator = df.count()
    return {col : df.where(F.col(col) == True).count() / denominator for col in cols}


def cardinalities(df, cols):
    from functools import reduce
    
    counts = df.agg(
        F.struct(*[F.countDistinct(F.col(c)).alias(c) for c in cols] + [F.count(F.col(cols[0])).alias('total')]).alias("results")
    ).select("results").collect()[0][0].asDict()
    counts.update({'total' : df.count()})
    return counts


def likely_unique(counts):
    total = counts["total"]
    return [k for (k, v) in counts.items() if k != "total" and abs(total - v) < total * 0.15]


def likely_categoricals(counts):
    total = counts["total"]
    return [k for (k, v) in counts.items() if v < total * 0.15 or v < 128]

def unique_values(df, cols):
    if eda_options['use_array_ops']:
        return unique_values_array(df, cols)
    else:   
        return unique_values_driver(df, cols)

def unique_values_array(df, cols):
    from functools import reduce
 
    counts = df.groupBy(
        F.lit(True).alias("drop_me")
    ).agg(
        *[F.array_sort(F.collect_set(F.col(c))).alias(c) for c in cols]
    ).drop("drop_me").cache()
    
    result = reduce(lambda l, r: l.unionAll(r), [counts.select(F.lit(c).alias("field"), F.col(c).alias("unique_vals")) for c in counts.columns]).collect()
    
    return dict([(r[0],r[1]) for r in result])


def unique_values_driver(df, cols):
    return { col : [v[0] for v in df.select(F.col(col).alias('value')).distinct().orderBy(F.col('value')).collect()] for col in cols}

def approx_ecdf(df, cols):
    from functools import reduce
    
    quantiles = [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]

    qs = df.approxQuantile(cols, quantiles, 0.01)
    
    result = dict(zip(cols, qs))
    return {c: dict(zip(quantiles, vs)) for (c, vs) in result.items()}


def gen_summary(df, output_prefix=""):
    summary = {}
    
    string_cols = []
    boolean_cols = []
    numeric_cols = []
    other_cols = []

    for field in df.schema.fields:
        if isinstance(field.dataType, T.StringType):
            string_cols.append(field.name)
        elif isinstance(field.dataType, T.BooleanType):
            boolean_cols.append(field.name)
        elif isnumeric(field.dataType):
            numeric_cols.append(field.name)
        else:
            other_cols.append(field.name)
    
    counts = cardinalities(df, string_cols)
    uniques = likely_unique(counts)
    categoricals = unique_values(df, likely_categoricals(counts))

    for span in [2,3,4,6,12]:
        thecube = df.cube("Churn", F.ceil(df.tenure / span).alias("%d_month_spans" % span), "gender", "Partner", "SeniorCitizen", "Contract", "PaperlessBilling", "PaymentMethod", F.ceil(F.log2(F.col("MonthlyCharges"))*10).alias("log_charges")).count()
        therollup = df.rollup("Churn", F.ceil(df.tenure / span).alias("%d_month_spans" % span), "SeniorCitizen", "Contract", "PaperlessBilling", "PaymentMethod", F.ceil(F.log2(F.col("MonthlyCharges"))*10).alias("log_charges")).agg(F.sum(F.col("TotalCharges")).alias("sum_charges"))
        thecube.write.mode("overwrite").parquet("%scube-%d.parquet" % (output_prefix, span))
        therollup.write.mode("overwrite").parquet("%srollup-%d.parquet" % (output_prefix, span))

    encoding_struct = {
        "categorical" : categoricals,
        "numeric" : numeric_cols + boolean_cols,
        "unique": uniques
    }
    
    summary["schema"] = df.schema.jsonValue()
    summary["ecdfs"] = approx_ecdf(df, numeric_cols)
    summary["true_percentage"] = percent_true(df, boolean_cols)
    summary["encoding"] = encoding_struct
    summary["distinct_customers"] = df.select(df.customerID).distinct().count()
    
    return summary

def losses_by_month(be):
    customer_lifetime_values = be.groupBy("customerID").sum("value").alias("value")
    return be.where(be.kind == "AccountTermination").join(customer_lifetime_values, "customerID").groupBy("month").sum("value").alias("value").sort("month").toPandas().to_json()

def output_reports(df, be=None, report_prefix=""):
    import json

    summary = gen_summary(df, report_prefix)

    if be is not None:
        summary["losses_by_month"] = losses_by_month(be)

    with open("%ssummary.json" % report_prefix, "w") as sf:
        json.dump(summary, sf)
    
    with open("%sencodings.json" % report_prefix, "w") as ef:
        json.dump(summary["encoding"], ef)
        
