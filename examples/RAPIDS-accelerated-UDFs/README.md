# RAPIDS Accelerated UDF Examples
This project contains sample implementations of RAPIDS accelerated user-defined functions.

## Spark Scala UDF Examples

- [URLDecode](./src/main/scala/com/nvidia/spark/rapids/udf/scala/URLDecode.scala)
  decodes URL-encoded strings using the
  [Java APIs of RAPIDS cudf](https://docs.rapids.ai/api/cudf-java/legacy)
- [URLEncode](./src/main/scala/com/nvidia/spark/rapids/udf/scala/URLEncode.scala)
  URL-encodes strings using the
  [Java APIs of RAPIDS cudf](https://docs.rapids.ai/api/cudf-java/legacy)

## Spark Java UDF Examples

- [URLDecode](./src/main/java/com/nvidia/spark/rapids/udf/java/URLDecode.java)
  decodes URL-encoded strings using the
  [Java APIs of RAPIDS cudf](https://docs.rapids.ai/api/cudf-java/legacy)
- [URLEncode](./src/main/java/com/nvidia/spark/rapids/udf/java/URLEncode.java)
  URL-encodes strings using the
  [Java APIs of RAPIDS cudf](https://docs.rapids.ai/api/cudf-java/legacy)
- [CosineSimilarity](./src/main/java/com/nvidia/spark/rapids/udf/java/CosineSimilarity.java)
  computes the [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity)
  between two float vectors using [native code](./src/main/cpp/src)

## Hive UDF Examples

- [URLDecode](./src/main/java/com/nvidia/spark/rapids/udf/hive/URLDecode.java)
  implements a Hive simple UDF using the
  [Java APIs of RAPIDS cudf](https://docs.rapids.ai/api/cudf-java/legacy)
  to decode URL-encoded strings
- [URLEncode](./src/main/java/com/nvidia/spark/rapids/udf/hive/URLEncode.java)
  implements a Hive generic UDF using the
  [Java APIs of RAPIDS cudf](https://docs.rapids.ai/api/cudf-java/legacy)
  to URL-encode strings
- [StringWordCount](./src/main/java/com/nvidia/spark/rapids/udf/hive/StringWordCount.java)
  implements a Hive simple UDF using
  [native code](./src/main/cpp/src) to count words in strings

## Building and run the tests without Native Code Examples
Some UDF examples use native code in their implementation.
Building the native code requires a libcudf build environment, so these
examples do not build by default.

### Prerequisites
Download Spark and set SPARK_HOME environment variable.
Refer to [Prerequisites](../../docs/get-started/xgboost-examples/on-prem-cluster/standalone-python.md#Prerequisites)  
Install python 3.8+, then install pytest, pyspark, sre_yield, findspark by using pip or conda.
For example:
```
pip install pytest
pip install pyspark
pip install sre_yield
pip install findspark
```

Run the following command to build and run tests
```bash
mvn clean package
./run_pyspark_from_build.sh -m "not rapids_udf_example_native"
```

## Building with Native Code Examples and run test cases
The `udf-native-examples` Maven profile
can be used to include the native UDF examples in the build, i.e.: specify
 `-Pudf-native-examples` on the `mvn` command-line.

### Creating a libcudf Build Environment
Building the native code requires a libcudf build environment.  
The `Dockerfile` in this directory can be used to setup a Docker image that
provides a libcudf build environment. This repository will either need to be
cloned or mounted into a container using that Docker image.
The `Dockerfile` contains build arguments to control the Linux version,
CUDA version, and other settings. See the top of the `Dockerfile` for details.

First install docker and [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)

Run the following commands to build and start a docker
```bash
cd spark-rapids-examples/examples/RAPIDS-accelerated-UDFs
docker build -t my-local:my-udf-example-ubuntu .
nvidia-docker run -it my-local:my-udf-example-ubuntu
```

### Build the udf-examples jar
In the docker, clone the code and compile.
```bash
git clone https://github.com/NVIDIA/spark-rapids-examples.git
cd spark-rapids-examples/examples/RAPIDS-accelerated-UDFs
mvn clean package -Pudf-native-examples
```
The building will spend some time like 1.5 hours.
Then the rapids-4-spark-udf-examples*.jar is generated under RAPIDS-accelerated-UDFs/target directory.

### Run all the examples including native examples in the docker
Download Spark and set SPARK_HOME environment variable.
Refer to [Prerequisites](../../docs/get-started/xgboost-examples/on-prem-cluster/standalone-python.md#Prerequisites)   
Set SPARK_HOME environment variable. 
```
export SPARK_HOME=path-to-spark
```
Install python 3.8+, then install pytest, pyspark, sre_yield, findspark by using pip or conda.
See above Prerequisites section
```
./run_pyspark_from_build.sh
```

## How to run the Native UDFs on Spark local mode
First finish the steps in "Building with Native Code Examples and run test cases" section, then do the following in the docker.

### Get jars from Maven Central
[cudf-21.12.0-cuda11.jar](https://repo1.maven.org/maven2/ai/rapids/cudf/21.12.0/cudf-21.12.0-cuda11.jar)   
[rapids-4-spark_2.12-21.12.0.jar](https://repo1.maven.org/maven2/com/nvidia/rapids-4-spark_2.12/21.12.0/rapids-4-spark_2.12-21.12.0.jar)

### Launch a local mode Spark

```bash
export SPARK_CUDF_JAR=path-to-cudf-jar
export SPARK_RAPIDS_PLUGIN_JAR=path-to-rapids-4-spark-jar
export SPARK_RAPIDS_UDF_EXAMPLES_JAR=path-to-udf-examples-jar

$SPARK_HOME/bin/pyspark --master local[*] \
--conf spark.executor.cores=6 \
--driver-memory 5G  \
--executor-memory 5G  \
--conf spark.executor.extraClassPath=${SPARK_CUDF_JAR}:${SPARK_RAPIDS_PLUGIN_JAR}:${SPARK_RAPIDS_UDF_EXAMPLES_JAR} \
--conf spark.driver.extraClassPath=${SPARK_CUDF_JAR}:${SPARK_RAPIDS_PLUGIN_JAR}:${SPARK_RAPIDS_UDF_EXAMPLES_JAR} \
--jars ${SPARK_CUDF_JAR},${SPARK_RAPIDS_PLUGIN_JAR},${SPARK_RAPIDS_UDF_EXAMPLES_JAR} \
--conf spark.plugins=com.nvidia.spark.SQLPlugin \
--conf spark.rapids.sql.enabled=true
```

### Test native based UDF

Input the following commands to test wordcount JIN UDF

```bash
from pyspark.sql.types import *
schema = StructType([
    StructField("c1", StringType()),
    StructField("c2", IntegerType()),
])
data = [
    ("s1",1),
    ("s2",2),
    ("s1",3),
    ("s2",3),
    ("s1",3),
]
df = spark.createDataFrame(
        SparkContext.getOrCreate().parallelize(data, numSlices=2),
        schema)
df.createOrReplaceTempView("tab")

spark.sql("CREATE TEMPORARY FUNCTION {} AS '{}'".format("wordcount", "com.nvidia.spark.rapids.udf.hive.StringWordCount"))
spark.sql("select wordcount(c1) from tab group by c1").show()
spark.sql("select wordcount(c1) from tab group by c1").explain()
```

Refer to [more Spark modes](../../docs/get-started/xgboost-examples/on-prem-cluster) to test against more Spark modes.