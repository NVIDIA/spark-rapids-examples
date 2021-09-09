# PCA example

This is an example of the GPU accelerated PCA algorithm running on Spark.


## API change

We describe main API changes for gpu accelerated algorithms:


```scala
val pca = new org.apache.spark.ml.feature.PCA()
  .setInputCol("feature")
  .setOutputCol("feature_value_3d")
  .setK(3)
  .useGemm(true) // or false, switch to use original BLAS bsr or cuBLAS gemm to compute covariance matrix
  .meanCentering(true) // or false, switch to do mean centering before computing covariance matrix
  .fit(vectorDf)
```

Note: only fit(train) process is accelerated.

## Build
_TODO: Add link to new repo for spark-rapids-ml_

User can build it directly in the _project root path_:
```
mvn clean package
```
Then `rapids-4-spark-ml_2.12-21.10.0-SNAPSHOT.jar` will be generated under `target` folder.

_Note_: This module contains both native and Java/Scala code. When compiling native code to get the essential library, `cmake`, `ninja`, and `gcc` are all required. 

TODO: Add jar link here

## How to use

Add the artifact jar to the Spark, for example:
```bash
$SPARK_HOME/bin/spark-shell --master $SPARK_MASTER \
 --driver-memory 20G \
 --executor-memory 30G \
 --conf spark.driver.maxResultSize=8G \
 --jars target/rapids-4-spark-ml_2.12-21.10.0-SNAPSHOT.jar \
 --conf spark.task.resource.gpu.amount=0.08 \
 --conf spark.executor.resource.gpu.amount=1 \
 --conf spark.executor.resource.gpu.discoveryScript=./getGpusResources.sh \
 --files ${SPARK_HOME}/examples/src/main/scripts/getGpusResources.sh
```

## Sample code

User can find sample scala code in `main.scala`. In the sample code, we will generate random data with 2048 feature dimensions. Then we use PCA to reduce number of features to 3.

Just copy the sample code into the shell laucnhed by the previous section and REPL will give out the algorithm results.