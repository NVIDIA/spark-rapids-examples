# Microbenchmark

## Introduction

The microbenchmark on [RAPIDS Accelerator For Apache Spark](https://nvidia.github.io/spark-rapids/) is to time the
conversion of incoming RDDs to and from a CachedBatch. Specifically to compare the performance of
ParquetCachedBatchSerializer to DefaultCachedBatchSerializer.

## Setup

### Build pcbs-benchmark

Build pcbs-benchmark project by

``` 
mvn clean package
```

Copy the spark-rapids jar that needs to be benchmarked to the lib folder in the project root and make sure to change the 
`PLUGIN_JAR` variable in `pcbs-benchmark.sh` to point to the right filename

### Start the Spark cluster

```
sbin/start-master.sh 
sbin/start-slave.sh <master-url>
```

## Run the benchmark

Set `SPARK_HOME` and `SPARK_MASTER_URL` and then the benchmark can be run by running the script file 
and passing the name of Parquet file to be read and cached e.g.

```
./pcbs-benchmark.sh <path to Parquet file>
```

The above statement will run this benchmark on the Parquet files in the given path folder. 

The benchmark can also be run on multiple files by modifying the `benchmark-multiple-files.sh` and adding files to the 
`FILE_NAMES` variable and then simply running the benchmarks by 

```
./benchmark-multiple-files.sh
```

The results of the run will be written to `cache-perf.txt` in the project root.
A sample of the file is listed below and should be self-explanatory

```
Reading file: /media/data/mortgage_parquet/20m/
Writing cache 5 times
Time taken for writes: Vector(2011, 1231, 1198, 1137, 1154)
Reading cache 10 times
Time taken for reads: Vector(505, 241, 214, 218, 211, 206, 217, 211, 207, 225)
Writing cache 5 times
Time taken for writes: Vector(9851, 8574, 8029, 8339, 8011)
Reading cache 10 times
Time taken for reads: Vector(9436, 3495, 3398, 3375, 3416, 3389, 3520, 3353, 3313, 3319)
PCBS
acc: true
Average write: 1346
Time taken for first read: 505
Average read (without first read): 216
acc: false
Average write: 8560
Time taken for first read: 9436
Average read (without first read): 3397
Reading file: /media/data/mortgage_parquet/20m/
Writing cache 5 times
Time taken for writes: Vector(9130, 7830, 7780, 7797, 7717)
Reading cache 10 times
Time taken for reads: Vector(6550, 711, 648, 634, 1207, 957, 652, 619, 680, 644)
Writing cache 5 times
Time taken for writes: Vector(8883, 7789, 7614, 7640, 7575)
Reading cache 10 times
Time taken for reads: Vector(6288, 613, 609, 601, 692, 614, 617, 628, 602, 599)
DefaultSerializer
acc: true
Average write: 8050
Time taken for first read: 6550
Average read (without first read): 750
acc: false
Average write: 7900
Time taken for first read: 6288
Average read (without first read): 619
```

### Note

If running benchmark for a single file the `cache-perf.txt` will need to be manually deleted if the old results are no 
longer needed. The benchmark purposefully appends to the file to keep the older results. This enables us to run the 
benchmark for multiple files and seeing all the results in the same place. 

