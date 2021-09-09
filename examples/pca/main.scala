import com.nvidia.spark.ml.feature.PCA
import org.apache.spark.ml.linalg._
import org.apache.spark.sql.functions._
val dim = 2048
val rows = 100000
val r = new scala.util.Random(0)

// generate dummy data
val dataDf = spark.createDataFrame(
  (0 until rows).map(_ => Tuple1(List.fill(dim)(r.nextDouble)))).withColumnRenamed("_1", "feature")

// use udf to meet ML algo input requirement: Vector input
val convertToVector = udf((array: Seq[Float]) => {
  Vectors.dense(array.map(_.toDouble).toArray)
})

val vectorDf = dataDf.withColumn("feature", convertToVector(col("feature")))

// use RAPIDS PCA class and enable cuBLAS gemm
val pcaGpu = new com.nvidia.spark.ml.feature.PCA().setInputCol("feature").setOutputCol("pca_features").setK(3).setUseGemm(true)
 
// train
val pcaModelGpu = spark.time(pcaGpu.fit(vectorDf))

// transform
pcaModelGpu.transform(vectorDf).select("pca_features").show(false)

// use original Spark ML PCA class
val pcaCpu = new org.apache.spark.ml.feature.PCA().setInputCol("feature").setOutputCol("pca_features").setK(3)

// train
val pcaModelCpu = spark.time(pcaCpu.fit(vectorDf))

// transform
pcaModelCpu.transform(vectorDf).select("pca_features").show(false)

