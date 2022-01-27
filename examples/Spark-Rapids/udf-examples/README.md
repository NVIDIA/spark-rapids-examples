# RAPIDS Accelerated UDF Examples
This project contains sample implementations of RAPIDS accelerated user-defined functions.

## Spark Scala UDF Examples

- [URLDecode](./src/main/scala/com/nvidia/spark/rapids/udf/scala/URLDecode.scala)
  decodes URL-encoded strings using the
  [Java APIs of RAPIDS cudf](https://docs.rapids.ai/api/cudf-java/stable)
- [URLEncode](./src/main/scala/com/nvidia/spark/rapids/udf/scala/URLEncode.scala)
  URL-encodes strings using the
  [Java APIs of RAPIDS cudf](https://docs.rapids.ai/api/cudf-java/stable)

## Spark Java UDF Examples

- [URLDecode](./src/main/java/com/nvidia/spark/rapids/udf/java/URLDecode.java)
  decodes URL-encoded strings using the
  [Java APIs of RAPIDS cudf](https://docs.rapids.ai/api/cudf-java/stable)
- [URLEncode](./src/main/java/com/nvidia/spark/rapids/udf/java/URLEncode.java)
  URL-encodes strings using the
  [Java APIs of RAPIDS cudf](https://docs.rapids.ai/api/cudf-java/stable)
- [CosineSimilarity](./src/main/java/com/nvidia/spark/rapids/udf/java/CosineSimilarity.java)
  computes the [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity)
  between two float vectors using [native code](./src/main/cpp/src)

## Hive UDF Examples

- [URLDecode](./src/main/java/com/nvidia/spark/rapids/udf/hive/URLDecode.java)
  implements a Hive simple UDF using the
  [Java APIs of RAPIDS cudf](https://docs.rapids.ai/api/cudf-java/stable)
  to decode URL-encoded strings
- [URLEncode](./src/main/java/com/nvidia/spark/rapids/udf/hive/URLEncode.java)
  implements a Hive generic UDF using the
  [Java APIs of RAPIDS cudf](https://docs.rapids.ai/api/cudf-java/stable)
  to URL-encode strings
- [StringWordCount](./src/main/java/com/nvidia/spark/rapids/udf/hive/StringWordCount.java)
  implements a Hive simple UDF using
  [native code](./src/main/cpp/src) to count words in strings


## Building the Native Code Examples

Some of the UDF examples use native code in their implementation.
Building the native code requires a libcudf build environment, so these
examples do not build by default. The `udf-native-examples` Maven profile
can be used to include the native UDF examples in the build, i.e.: specify
 `-Pudf-native-examples` on the `mvn` command-line.

### Creating a libcudf Build Environment

The `Dockerfile` in this directory can be used to setup a Docker image that
provides a libcudf build environment. This repository will either need to be
cloned or mounted into a container using that Docker image.
The `Dockerfile` contains build arguments to control the Linux version,
CUDA version, and other settings. See the top of the `Dockerfile` for details.

Run the following commands to build and start a docker
```bash
cd spark-rapids-examples/examples/Spark-Rapids/udf-examples
docker build -t my-local:my-udf-example-ubuntu .
docker run -it my-local:my-udf-example-ubuntu
```

### Build the udf-example jar
In the docker, clone the code and compile.

```bash
git clone https://github.com/NVIDIA/spark-rapids-examples.git
cd spark-rapids-examples/examples/Spark-Rapids/udf-examples
mvn clean package -Pudf-native-examples
```
Then the udf-example*.jar is generated under udf-examples/target directory.

## How to run the Native UDF on Spark local mode
After built the Native Code Examples, do the following

### Prerequisites
Download Spark and set SPARK_HOME environment variable.
Refer to [Prerequisites](../../../docs/get-started/xgboost-examples/on-prem-cluster/standalone-python.md#Prerequisites)

### Get jars from Maven Central
[cudf-21.12.0-cuda11.jar](https://repo1.maven.org/maven2/ai/rapids/cudf/21.12.0/cudf-21.12.0-cuda11.jar)   
[rapids-4-spark_2.12-21.12.0.jar](https://repo1.maven.org/maven2/com/nvidia/rapids-4-spark_2.12/21.12.0/rapids-4-spark_2.12-21.12.0.jar)

### Launch a local mode Spark

```bash
export SPARK_CUDF_JAR=path-to-cudf-jar
export SPARK_RAPIDS_PLUGIN_JAR=path-to-rapids-4-spark-jar
export SPARK_RAPIDS_UDF_EXAMPLES_JAR=path-to-udf-example-jar

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

Refer to [more Spark modes](../../../docs/get-started/xgboost-examples/on-prem-cluster) to test against more Spark modes.