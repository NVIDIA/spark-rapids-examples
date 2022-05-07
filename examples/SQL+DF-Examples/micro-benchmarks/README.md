# Microbenchmark 

## Introduction
The microbenchmark on [RAPIDS Accelerator For Apache Spark](https://nvidia.github.io/spark-rapids/) is to identify, 
test and analyze the best queries which can be accelerated on the GPU. 
The queries are based on several tables in [TPC-DS](http://www.tpc.org/tpcds/) parquet format,
so that similar speedups can be reproducible by others.
The microbenchmark includes commonly used Spark SQL operations such as expand, hash aggregate, windowing, and cross joins,
and runs the same queries in CPU mode and GPU mode. Some queries will involve data skew.
Each of them is highly [tuned](https://nvidia.github.io/spark-rapids/docs/tuning-guide.html) and works with the optimal configuration
on an 8 nodes Spark standalone cluster which with 128 CPU cores and 1 A100 GPU on each node. 

## Dataset
You can generate the parquet format dataset using this [Databricks Tool](https://github.com/databricks/spark-sql-perf).
All the queries are running on the SF3000(Scale Factors 3TB) dataset. You can generate it with the following command:
```
# Assuming your platform is Linux
# Install sbt
echo "deb https://repo.scala-sbt.org/scalasbt/debian all main" | sudo tee /etc/apt/sources.list.d/sbt.list
echo "deb https://repo.scala-sbt.org/scalasbt/debian /" | sudo tee /etc/apt/sources.list.d/sbt_old.list
curl -sL "https://keyserver.ubuntu.com/pks/lookup?op=get&search=0x2EE0EA64E40A89B84B2DF73499E82A75642AC823" | sudo apt-key add
sudo apt-get update
sudo apt-get install sbt

# Install jdk 
sudo apt-get install openjdk-8-jdk

# clone related repos
git clone https://github.com/databricks/spark-sql-perf.git
git clone https://github.com/databricks/tpcds-kit.git

# build 
cd tpcds-kit/tools
make OS=LINUX

sbt "test:runMain com.databricks.spark.sql.perf.tpcds.GenTPCDSData -d /databricks-tpcds-kit-path -s 3000G -l /your-dataset-path -f parquet"
```

## Note
You will see the [RAPIDS Accelerator For Apache Spark](https://nvidia.github.io/spark-rapids/) can give speedups of up to 10x over the CPU, and in some cases up to 80x.
It is easy to compare the [microbenchmarks on CPU](/examples/SQL+DF-Examples/micro-benchmarksmicro-benchmarks/notebooks/micro-benchmarks-cpu.ipynb) and [GPU](/examples/SQL+DF-Examples/micro-benchmarksmicro-benchmarks/notebooks/micro-benchmarks-gpu.ipynb) side by side.
You can see some queries are faster in the second time, it can be caused by many reasons such as JVM JIT or initialization overhead or caching input data in the OS page cache, etc.
You can get a clear and visual impression of the improved performance with or without the benefits of post-running.
The improved performance is influenced by many conditions, including the dataset's scale factors or the GPU card.
If the application ran for too long or even failed, you can run the queries on a smaller dataset.