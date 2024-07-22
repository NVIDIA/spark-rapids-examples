# Microbenchmark

Standard industry benchmarks are a great way to measure performance over 
a period of time but another barometer to measure performance is to measure
performance of common operators that are used in the data preprocessing stage or in data analytics.
The microbenchmark notebook in this repo uses five such queries in the chart shown below:

- **Count Distinct**: a function used to estimate the number of unique page views or 
  unique customers visiting an e-commerce site.
- **Window**: a critical operator necessary for preprocessing components in analyzing
  timestamped event data in marketing or financial industry.
- **Intersect**: an operator used to remove duplicates in a dataframe.
- **Cross-join**: A common use for a cross join is to obtain all combinations of items.
- **Hash-join**: Joining two tables together by matching rows based on a common column.

These queries were run on a standard eight-nodes CPU cluster with 2 CPU (128 cores),
512GB memory and 1xA100 GPUs per node. The dataset used was of size 3TB with multiple different data types.
The queries are based on several tables in NDS parquet format with Decimal. 
These four queries show not only performance and cost benefits but also the range of
speed-up (27x to 1.5x) varies depending on compute intensity. 
These queries vary in compute and network utilization similar to a practical use case in
data preprocessing.To test these queries, you can generate the parquet format dataset using
this NDS dataset generator tool. All the queries are running on the SF3000(Scale Factor 3000) dataset.
You can generate it with the following command:
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

![microbenchmark-speedup](/docs/img/guides/microbm.png)
