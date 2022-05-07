# Microbenchmark 

## Introduction
The microbenchmark on [RAPIDS Accelerator For Apache Spark](https://nvidia.github.io/spark-rapids/) is to identify, 
test and analyze the best queries which can be accelerated on the GPU. 
The queries are based on several tables in [TPC-DS](http://www.tpc.org/tpcds/) parquet format with Double replacing Decimal,
so that similar speedups can be reproducible by others.
The microbenchmark includes commonly used Spark SQL operations such as expand, hash aggregate, windowing, and cross joins,
and runs the same queries in CPU mode and GPU mode. Some queries will involve data skew.
Each of them is highly [tuned](https://nvidia.github.io/spark-rapids/docs/tuning-guide.html) and works with the optimal configuration
on an 8 nodes Spark standalone cluster which with 128 CPU cores and 1 A100 GPU on each node. 

## Dataset
You can generate the parquet format dataset using this [Databricks Tool](https://github.com/databricks/spark-sql-perf).
All the queries are running on the SF3000(Scale Factors 3TB) dataset. You can generate it with the following command:
```
build/sbt "test:runMain com.databricks.spark.sql.perf.tpcds.GenTPCDSData -d /databricks-tpcds-kit-path -s 3000G -l /your-dataset-path -f parquet"
```

## Note
You will see the [RAPIDS Accelerator For Apache Spark](https://nvidia.github.io/spark-rapids/) can give speedups of up to 10x over the CPU, and in some cases up to 80x.
It is easy to compare the [microbenchmarks on CPU](/examples/SQL+DF-Examples/micro-benchmarksmicro-benchmarks/notebooks/micro-benchmarks-cpu.ipynb) and [GPU](/examples/SQL+DF-Examples/micro-benchmarksmicro-benchmarks/notebooks/micro-benchmarks-gpu.ipynb) side by side.
You can see some queries are faster in the second time, it can be caused by many reasons such as JVM JIT or initialization overhead or caching input data in the OS page cache, etc.
You can get a clear and visual impression of the improved performance with or without the benefits of post-running.
The improved performance is influenced by many conditions, including the dataset's scale factors or the GPU card.
If the application ran for too long or even failed, you can run the queries on a smaller dataset.