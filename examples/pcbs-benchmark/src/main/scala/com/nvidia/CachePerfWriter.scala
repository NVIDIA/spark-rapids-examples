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
package com.nvidia

import java.io.{BufferedWriter, File, FileWriter, IOException}

/**
 * @author Raza Jafri
 */
object CachePerfWriter {
  val file = new File("cache-perf.txt")
  lazy val bw = new BufferedWriter(new FileWriter(file, true))

  @throws[IOException]
  def appendLine(a: String): Unit = {
    bw.append(s"$a\n")
  }

  def appendTimes(
      acc: Boolean,
      write: Long,
      firstRead: Long,
      readAvg: Long): Unit ={
    appendLine(s"acc: $acc")
    appendLine("Average write: " + write)
    appendLine("Time taken for frst read: " + firstRead)
    appendLine("Average read (without first read): " + readAvg)
  }

  def close(): Unit = {
    bw.close()
  }
}
