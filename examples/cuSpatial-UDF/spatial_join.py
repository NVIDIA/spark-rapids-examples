#
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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
#

import sys
import time
from pyspark.sql import SparkSession


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise Exception("Requires a data path.")
    inputPath = sys.argv[1]
    outputPath = sys.argv[2]

    spark = SparkSession.builder.getOrCreate()
    # register the udf and set its parameters via the runtime config
    spark.udf.registerJavaFunction("point_in_polygon", "com.nvidia.spark.rapids.udf.PointInPolygon", None)
    spark.conf.set("spark.cuspatial.sql.udf.shapeFileName", "taxi_zones.shp")

    # read the points data
    df = spark.read.parquet(inputPath)
    # null row is not supported yet by the UDF, filter out them first.
    df = df.filter("x is not NULL and y is not NULL")

    # test func start
    df = df.selectExpr('x', 'y', 'point_in_polygon(x, y) as point_in_polygon')
    # test func end

    # trigger the test
    begin = time.time()
    df.write.mode("overwrite").parquet(outputPath)
    end = time.time()
    print("==> It took {} s".format(round(end-begin, 2)))
    spark.stop()
