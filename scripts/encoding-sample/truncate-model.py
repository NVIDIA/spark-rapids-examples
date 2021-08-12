import sys

from pyspark.sql import SparkSession
from pyspark.sql.functions import *

(SparkSession
    .builder
    .getOrCreate()
    .read
    .csv(sys.argv[1])
    .withColumn('_c1', format_string('%.6f', col('_c1').cast('float')))
    .withColumn('_c1', when(col('_c1') == '0.000000', lit('0.0')).otherwise(col('_c1')))
    .withColumn('_c1', when(col('_c1') == '1.000000', lit('1.0')).otherwise(col('_c1')))
    .repartition(1)
    .write
    .option('nullValue', None)
    .csv(sys.argv[2]))
