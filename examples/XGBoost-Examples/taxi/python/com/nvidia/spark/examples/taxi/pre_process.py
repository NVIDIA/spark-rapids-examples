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
import math

from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.functions import col


def pre_process(data_frame):
    processes = [
        drop_useless,
        encode_categories,
        fill_na,
        remove_invalid,
        convert_datetime,
        add_h_distance,
    ]
    for process in processes:
        data_frame = process(data_frame)
    return data_frame

def drop_useless(data_frame):
    return data_frame.drop(
        'dropoff_datetime',
        'payment_type',
        'surcharge',
        'mta_tax',
        'tip_amount',
        'tolls_amount',
        'total_amount')

def encode_categories(data_frame):
    categories = [ 'vendor_id', 'rate_code', 'store_and_fwd_flag' ]
    for category in categories:
        data_frame = data_frame.withColumn(category, hash(col(category)))
    return data_frame.withColumnRenamed("store_and_fwd_flag", "store_and_fwd")

def fill_na(data_frame):
    return data_frame.fillna(-1)

def remove_invalid(data_frame):
    conditions = [
        ( 'fare_amount', 0, 500 ),
        ( 'passenger_count', 0, 6 ),
        ( 'pickup_longitude', -75, -73 ),
        ( 'dropoff_longitude', -75, -73 ),
        ( 'pickup_latitude', 40, 42 ),
        ( 'dropoff_latitude', 40, 42 ),
    ]
    for column, min, max in conditions:
        data_frame = data_frame.filter('{} > {} and {} < {}'.format(column, min, column, max))
    return data_frame

def convert_datetime(data_frame):
    datetime = col('pickup_datetime')
    return (data_frame
        .withColumn('pickup_datetime', to_timestamp(datetime))
        .withColumn('year', year(datetime))
        .withColumn('month', month(datetime))
        .withColumn('day', dayofmonth(datetime))
        .withColumn('day_of_week', dayofweek(datetime))
        .withColumn(
            'is_weekend',
            col('day_of_week').isin(1, 7).cast(IntegerType()))  # 1: Sunday, 7: Saturday
        .withColumn('hour', hour(datetime))
        .drop('pickup_datetime'))

def add_h_distance(data_frame):
    p = math.pi / 180
    lat1 = col('pickup_latitude')
    lon1 = col('pickup_longitude')
    lat2 = col('dropoff_latitude')
    lon2 = col('dropoff_longitude')
    internal_value = (0.5
        - cos((lat2 - lat1) * p) / 2
        + cos(lat1 * p) * cos(lat2 * p) * (1 - cos((lon2 - lon1) * p)) / 2)
    h_distance = 12734 * asin(sqrt(internal_value))
    return data_frame.withColumn('h_distance', h_distance)
