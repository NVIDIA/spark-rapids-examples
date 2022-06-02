#
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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
from com.nvidia.spark.examples.mortgage.consts import *
from pyspark.sql.functions import *
from pyspark.sql.types import *
from sys import exit

get_quarter = udf(lambda path: path.split(r'.')[0].split('_')[-1], StringType())
standardize_name = udf(lambda name: name_mapping.get(name), StringType())

def load_data(spark, paths, schema, args, extra_csv_opts={}):
    reader = (spark
        .read
        .format(args.format)
        .option('asFloats', args.asFloats)
        .option('maxRowsPerChunk', args.maxRowsPerChunk))
    if args.format == 'csv':
        (reader
            .schema(schema)
            .option('delimiter', '|')
            .option('header', args.hasHeader))
        for k, v in extra_csv_opts.items():
            reader.option(k, v)
    return reader.load(paths)

def prepare_performance(spark, args):
    extra_csv_options = {
        'nullValue': '',
        'parserLib': 'univocity',
    }
    paths = extract_paths(args.dataPaths, 'perf::')
    performance = (load_data(spark, paths, performance_schema, args, extra_csv_options)
        .withColumn('quarter', get_quarter(input_file_name()))
        .withColumn('timestamp', to_date(col('monthly_reporting_period'), 'MM/dd/yyyy'))
        .withColumn('timestamp_year', year(col('timestamp')))
        .withColumn('timestamp_month', month(col('timestamp'))))

    aggregation = (performance
        .select(
            'quarter',
            'loan_id',
            'current_loan_delinquency_status',
            when(col('current_loan_delinquency_status') >= 1, col('timestamp'))
                .alias('delinquency_30'),
            when(col('current_loan_delinquency_status') >= 3, col('timestamp'))
                .alias('delinquency_90'),
            when(col('current_loan_delinquency_status') >= 6, col('timestamp'))
                .alias('delinquency_180'))
        .groupBy('quarter', 'loan_id')
        .agg(
            max('current_loan_delinquency_status').alias('delinquency_12'),
            min('delinquency_30').alias('delinquency_30'),
            min('delinquency_90').alias('delinquency_90'),
            min('delinquency_180').alias('delinquency_180'))
        .select(
            'quarter',
            'loan_id',
            (col('delinquency_12') >= 1).alias('ever_30'),
            (col('delinquency_12') >= 3).alias('ever_90'),
            (col('delinquency_12') >= 6).alias('ever_180'),
            'delinquency_30',
            'delinquency_90',
            'delinquency_180'))

    months = spark.createDataFrame(range(12), IntegerType()).withColumnRenamed('value', 'month_y')
    to_join = (performance
        .select(
            'quarter',
            'loan_id',
            'timestamp_year',
            'timestamp_month',
            col('current_loan_delinquency_status').alias('delinquency_12'),
            col('current_actual_upb').alias('upb_12'))
        .join(aggregation, ['loan_id', 'quarter'], 'left_outer')
        .crossJoin(months)
        .select(
            'quarter',
            floor(
                (col('timestamp_year') * 12 + col('timestamp_month') - 24000 - col('month_y')) / 12
            ).alias('josh_mody_n'),
            'ever_30',
            'ever_90',
            'ever_180',
            'delinquency_30',
            'delinquency_90',
            'delinquency_180',
            'loan_id',
            'month_y',
            'delinquency_12',
            'upb_12')
        .groupBy(
            'quarter',
            'loan_id',
            'josh_mody_n',
            'ever_30',
            'ever_90',
            'ever_180',
            'delinquency_30',
            'delinquency_90',
            'delinquency_180',
            'month_y')
        .agg(
            max('delinquency_12').alias('delinquency_12'),
            min('upb_12').alias('upb_12'))
        .withColumn(
            'timestamp_year',
            floor((24000 + (col('josh_mody_n') * 12) + (col('month_y') - 1)) / 12))
        .withColumn(
            'timestamp_month_tmp',
            (24000 + (col('josh_mody_n') * 12) + col('month_y')) % 12)
        .withColumn(
            'timestamp_month',
            when(col('timestamp_month_tmp') == 0, 12).otherwise(col('timestamp_month_tmp')))
        .withColumn(
            'delinquency_12',
            ((col('delinquency_12') > 3).cast('int') + (col('upb_12') == 0).cast('int')))
        .drop('timestamp_month_tmp', 'josh_mody_n', 'month_y'))

    return (performance
        .join(to_join, ['quarter', 'loan_id', 'timestamp_year', 'timestamp_month'], 'left')
        .drop('timestamp_year', 'timestamp_month'))

def prepare_acquisition(spark, args):
    return (load_data(spark, extract_paths(args.dataPaths, 'acq::'), acquisition_schema, args)
        .withColumn('quarter', get_quarter(input_file_name()))
        .withColumn('seller_name', standardize_name(col('seller_name'))))

def extract_paths(paths, prefix):
    results = [ path[len(prefix):] for path in paths if path.startswith(prefix) ]
    if not results:
        print('-' * 80)
        print('Usage: {} data path required'.format(prefix))
        exit(1)
    return results

def etl(spark, args):
    performance = prepare_performance(spark, args)
    acquisition = prepare_acquisition(spark, args)
    return (performance
        .join(acquisition, ['loan_id', 'quarter'], 'left_outer')
        .select(
            [(md5(col(x)) % 100).alias(x) for x in categorical_columns]
            + [col(x) for x in numeric_columns])
        .withColumn('delinquency_12', when(col('delinquency_12') > 0, 1).otherwise(0))
        .na
        .fill(0))

