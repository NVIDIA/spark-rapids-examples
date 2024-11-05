#
# Copyright (c) 2019-2022, NVIDIA CORPORATION. All rights reserved.
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

from pyspark.sql.types import *

label = 'fare_amount'

raw_schema = StructType([
    StructField('vendor_id', StringType()),
    StructField('pickup_datetime', StringType()),
    StructField('dropoff_datetime', StringType()),
    StructField('passenger_count', IntegerType()),
    StructField('trip_distance', DoubleType()),
    StructField('pickup_longitude', DoubleType()),
    StructField('pickup_latitude', DoubleType()),
    StructField('rate_code', StringType()),
    StructField('store_and_fwd_flag', StringType()),
    StructField('dropoff_longitude', DoubleType()),
    StructField('dropoff_latitude', DoubleType()),
    StructField('payment_type', StringType()),
    StructField(label, DoubleType()),
    StructField('surcharge', DoubleType()),
    StructField('mta_tax', DoubleType()),
    StructField('tip_amount', DoubleType()),
    StructField('tolls_amount', DoubleType()),
    StructField('total_amount', DoubleType()),
])

final_schema = StructType([
    StructField('vendor_id', FloatType()),
    StructField('passenger_count', FloatType()),
    StructField('trip_distance', FloatType()),
    StructField('pickup_longitude', FloatType()),
    StructField('pickup_latitude', FloatType()),
    StructField('rate_code', FloatType()),
    StructField('store_and_fwd', FloatType()),
    StructField('dropoff_longitude', FloatType()),
    StructField('dropoff_latitude', FloatType()),
    StructField(label, FloatType()),
    StructField('hour', FloatType()),
    StructField('year', IntegerType()),
    StructField('month', IntegerType()),
    StructField('day', FloatType()),
    StructField('day_of_week', FloatType()),
    StructField('is_weekend', FloatType()),
])
