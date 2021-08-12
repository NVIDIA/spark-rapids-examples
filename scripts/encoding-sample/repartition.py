# Note: Plase modify the data source options for your case.

import sys

from pyspark.sql import SparkSession

(SparkSession
    .builder
    .getOrCreate()
    .read
    .option('sep', '\t')
    .csv(sys.argv[1])
    .repartition(int(sys.argv[3]))
    .write
    .option('sep', '\t')
    .option('nullValue', None)
    .csv(sys.argv[2]))
