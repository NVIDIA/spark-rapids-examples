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

package com.nvidia.spark.rapids.udf;

import ai.rapids.cudf.ColumnVector;
import com.nvidia.spark.RapidsUDF;
import org.apache.spark.SparkFiles;
import org.apache.spark.sql.api.java.UDF2;
import org.apache.spark.sql.internal.SQLConf;

import java.io.File;
import java.util.List;

/**
 * A Spark Java UDF that computes what the `spatial_join` test does here.
 * https://github.com/zhangjianting/cuspatial_benchmark_nyctaxi/blob/main/python/spatial_join.py
 */
public class PointInPolygon implements UDF2<Double, Double, List<Integer>>, RapidsUDF {

  private String shapeFile;
  private boolean localPathParsed;

  public PointInPolygon() {}

  public PointInPolygon(String shapeFileName) {
    this.shapeFile = shapeFileName;
  }

  /** Row-by-row implementation that executes on the CPU */
  @Override
  public List<Integer> call(Double x, Double y) {
    // not supported yet
    throw new UnsupportedOperationException();
  }

  // Share the polygon columns across UDF instances, since each task will create a UDF.
  private final static ColumnVector[] polygons = new ColumnVector[4];
  private static int refCount;

  private void ensureShapeFile() {
    // Read the config from SQLConf to support runtime updating
    String newShapeName = SQLConf.get().getConfString("spark.cuspatial.sql.udf.shapeFileName", null);
    boolean validNewName = newShapeName != null && !newShapeName.equals(shapeFile);
    // Each task has a different UDF instance so no need to sync when operating object members.
    if (!localPathParsed || validNewName) {
      if (validNewName) {
        // update to the latest
        shapeFile = newShapeName;
      }
      if (shapeFile == null || shapeFile.isEmpty()) {
        throw new RuntimeException("Shape file name is missing");
      }
      // Get the local path of the downloaded file on each work node.
      shapeFile = SparkFiles.get(new File(shapeFile).getName());
      localPathParsed = true;
    }

    // load the shape data when needed
    synchronized (polygons) {
      if (refCount == 0) {
        long[] ret = readPolygon(shapeFile);
        try {
          assert ret.length == polygons.length;
          // Table is not applicable here because the columns may have different row numbers.
          // So need to cache it as an array.
          for (int i = 0; i < polygons.length; i++) {
            polygons[i] = new ColumnVector(ret[i]);
          }
        } catch (Throwable t) {
          for (int i = 0; i < polygons.length; i++) {
            if (polygons[i] != null) {
              polygons[i].close();
              polygons[i] = null;
            } else if (ret[i] != 0){
              deleteCudfColumn(ret[i]);
            }
          }
          throw t;
        }
      } // end of 'if'
      // "refCount < 0" can not happen because this method is private.
      // the order of increasing/decreasing the reference is managed internally.
      refCount ++;
    }
  }

  private void releaseShapeData() {
    synchronized (polygons) {
      refCount --;
      if (refCount == 0) {
        for (int i = 0; i < polygons.length; i++) {
          if (polygons[i] != null) {
            polygons[i].close();
            polygons[i] = null;
          }
        }
      }
      // "refCount < 0" can not happen because this method is private.
      // the order of increasing/decreasing the reference is managed internally.
    }
  }

  /** Columnar implementation that processes data on the GPU */
  @Override
  public ColumnVector evaluateColumnar(int numRows, ColumnVector... args) {
    if (args.length != 2) {
      throw new IllegalArgumentException("Unexpected argument count: " + args.length +
          ", expects 2 for (x, y)");
    }

    // Load the native code if it has not been already loaded. This is done here
    // rather than in a static code block because the driver may not have the
    // required CUDA environment.
    UDFNativeLoader.ensureLoaded();
    ensureShapeFile();

    try {
      return new ColumnVector(pointInPolygon(args[0].getNativeView(), args[1].getNativeView(),
          polygons[0].getNativeView(), polygons[1].getNativeView(),
          polygons[2].getNativeView(), polygons[3].getNativeView()));
    } finally {
      // Now try to open/close the shape file per call. This can avoid the duplicate readings
      // when multiple tasks in the same executors run in parallel. But each batch in the
      // same task still executes a new reading.
      // A better solution is to try to open/close the shape file per task, then it can also
      // avoid duplicate readings per batch in the same task.
      //
      // All of this is to figure out a proper time to close the columns of the shape data.
      releaseShapeData();
    }
  }

  private static native void deleteCudfColumn(long cudfColumnHandle);

  /**
   * read the polygon shape file as array of 4 columns representing one or more polygons:
   *   [0]-INT32:   beginning index of the first ring in each polygon
   *   [1]-INT32:   beginning index of the first point in each ring
   *   [2]-FLOAT64: x component of polygon points
   *   [3]-FLOAT64: y component of polygon points
   */
  private static native long[] readPolygon(String shapeFile);

  /** Native implementation that computes on the GPU */
  private static native long pointInPolygon(long xColView, long yColView,
                                            long plyFPosView, long plyRPosView,
                                            long plyXView, long plyYView);
}
