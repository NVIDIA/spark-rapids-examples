/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.sql.execution.columnar

import java.lang.reflect.Method
import java.util.{Locale, TimeZone}

import com.nvidia.spark.ParquetCachedBatchSerializer
import com.nvidia.spark.rapids.ParquetCachedBatch
import com.nvidia.CachePerfWriter

import org.apache.spark.{sql, TaskContext}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.Attribute
import org.apache.spark.sql.columnar.{CachedBatch, CachedBatchSerializer}
import org.apache.spark.sql.execution.SparkPlan
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.rapids.GpuInMemoryTableScanExec
import org.apache.spark.sql.vectorized.ColumnarBatch
import org.apache.spark.storage.StorageLevel
import org.apache.spark.storage.StorageLevel.MEMORY_ONLY

class TestCachedBatchSerializer(
    useCompression: Boolean,
    batchSize: Int) extends DefaultCachedBatchSerializer {

  override def convertInternalRowToCachedBatch(input: RDD[InternalRow],
      schema: Seq[Attribute],
      storageLevel: StorageLevel,
      conf: SQLConf): RDD[CachedBatch] = {
    convertForCacheInternal(input, schema, batchSize, useCompression)
  }
}

private case class CloseableColumnBatchIterator(iter: Iterator[ColumnarBatch]) extends
    Iterator[ColumnarBatch] {
  var cb: ColumnarBatch = _

  private def closeCurrentBatch(): Unit = {
    if (cb != null) {
      cb.close()
      cb = null
    }
  }

  TaskContext.get().addTaskCompletionListener[Unit]((_: TaskContext) => {
    closeCurrentBatch()
  })

  override def hasNext: Boolean = iter.hasNext

  override def next(): ColumnarBatch = {
    closeCurrentBatch()
    cb = iter.next()
    cb
  }
}

object Util {
  def createSparkSession(): SparkSession = {
    // Timezone is fixed to UTC to allow timestamps to work by default
    TimeZone.setDefault(TimeZone.getTimeZone("UTC"))
    // Add Locale setting
    Locale.setDefault(Locale.US)
    val sparkMasterUrl = System.getenv("SPARK_MASTER_URL")
    val builder = SparkSession.builder()
        .master(sparkMasterUrl)
        .appName("pcbs perf (scala)")
    builder.getOrCreate()
  }

  def readWriteCache(
      acc: Boolean,
      spark: SparkSession,
      ser: CachedBatchSerializer,
      func: sql.DataFrame => (Method, Seq[SparkPlan]),
      verifyFunc: CachedBatch => Any,
      query: SparkSession => sql.DataFrame): (Long, Long, Long) = {
    CachePerfWriter.appendLine("Writing cache 5 times")
    val writes = for (_ <- 0 until 5) yield {
      val df = query(spark).cache()
      val storageLevel = MEMORY_ONLY
      val plan = spark.sessionState.executePlan(df.logicalPlan).sparkPlan
      val relation = InMemoryRelation(ser, storageLevel, plan, None, df.logicalPlan)
      val start = System.currentTimeMillis()
      val cb = relation.cacheBuilder.cachedColumnBuffers.first()
      val defaWriteTime = System.currentTimeMillis() - start
      verifyFunc(cb)
      df.unpersist(true)
      defaWriteTime
    }
    CachePerfWriter.appendLine(s"Time taken for writes: $writes")
    val defaWriteTime = writes.sum / 5

    val df = query(spark).cache()
    val storageLevel = MEMORY_ONLY
    val plan = spark.sessionState.executePlan(df.logicalPlan).sparkPlan
    val relation = InMemoryRelation(ser, storageLevel, plan, None, df.logicalPlan)
    relation.cacheBuilder.cachedColumnBuffers
    val (doExecuteMethod, inMemoryScans) = func(df)
    CachePerfWriter.appendLine("Reading cache 10 times")
    val reads = for (_ <- 0 until 10) yield {
      val start = System.currentTimeMillis()
      val inMemoryScan = inMemoryScans.head
      val rdd = doExecuteMethod.invoke(inMemoryScan).asInstanceOf[RDD[ColumnarBatch]]
      if (ser.isInstanceOf[ParquetCachedBatchSerializer] && acc) {
        rdd.mapPartitions(iter => CloseableColumnBatchIterator(iter)).count()
        rdd.foreach {
          cb => cb.close()
        }
      } else {
        rdd.count()
      }
      System.currentTimeMillis() - start
    }
    CachePerfWriter.appendLine(s"Time taken for reads: $reads")
    val defaReadTime = reads.slice(1, reads.length).sum / 9
    df.unpersist()
    (defaWriteTime, defaReadTime, reads(0))
  }

  def runDefaInternal[T](
      query: SparkSession => DataFrame,
      acc: Boolean,
      ser: CachedBatchSerializer): (Boolean, Long, Long, Long) = {
    val spark = createSparkSession()
    if (acc) {
      spark.conf.set("spark.rapids.sql.enabled", "true")
    } else {
      spark.conf.set("spark.rapids.sql.enabled", "false")
    }
    val (defaWriteTime, defaReadTime, firstRead) =
      readWriteCache(acc, spark, ser, { df =>
        val doExecuteMethod =
          classOf[InMemoryTableScanExec].getDeclaredMethod("doExecute")
        doExecuteMethod.setAccessible(true)
        val inMemScans = df.queryExecution.executedPlan.collect {
          case m: InMemoryTableScanExec => m
        }
        (doExecuteMethod, inMemScans)
      }, cb =>
        cb match {
          case _: T =>
          case other => throw new IllegalStateException(s"Unexpected cached batch type: ${other.getClass.getName}")
        }, query )
    (acc, defaWriteTime, firstRead, defaReadTime)
  }

  def runPcbsInternal(query: SparkSession => DataFrame, acc: Boolean): (Boolean, Long, Long, Long) = {
    val spark = createSparkSession()
    if (acc) {
      spark.conf.set("spark.rapids.sql.enabled", "true")
    } else {
      spark.conf.set("spark.rapids.sql.enabled", "false")
    }
    val (pcbsWriteTime, pcbsReadTime, firstRead) =
      readWriteCache(acc, spark, new ParquetCachedBatchSerializer(), { df =>
        val doExecuteMethod = classOf[GpuInMemoryTableScanExec].getDeclaredMethod("doExecuteColumnar")
        doExecuteMethod.setAccessible(true)
        val inMemScans = df.queryExecution.executedPlan.collect {
          case g: GpuInMemoryTableScanExec => g
          case m: InMemoryTableScanExec => m
        }
        (doExecuteMethod, inMemScans)
      }, cb =>
        cb match {
          case _: ParquetCachedBatch =>
          case other => throw new IllegalStateException(s"Unexpected cached batch type: ${other.getClass.getName}")
        }, query)

    (acc, pcbsWriteTime, firstRead, pcbsReadTime)
  }

  def runPcbs(query: SparkSession => DataFrame): Unit = {
    val (acc, write, firstRead, readAvg) = runPcbsInternal(query, true)
    val pcbs = new ParquetCachedBatchSerializer()
    val (acc0, write0, firstRead0, readAvg0) = runDefaInternal[ParquetCachedBatch](query, false, pcbs)
    CachePerfWriter.appendLine("PCBS")
    CachePerfWriter.appendTimes(acc, write, firstRead, readAvg)
    CachePerfWriter.appendTimes(acc0, write0, firstRead0, readAvg0)
  }

  def runDefa(query: SparkSession => DataFrame): Unit = {
    val ser = new TestCachedBatchSerializer(useCompression = true, 10000)
    val (acc, write, firstRead, readAvg) = runDefaInternal[DefaultCachedBatch](query, true, ser)
    val (acc0, write0, firstRead0, readAvg0) = runDefaInternal[DefaultCachedBatch](query, false, ser)

    CachePerfWriter.appendLine("DefaultSerializer")
    CachePerfWriter.appendTimes(acc, write, firstRead, readAvg)
    CachePerfWriter.appendTimes(acc0, write0, firstRead0, readAvg0)
  }
}
