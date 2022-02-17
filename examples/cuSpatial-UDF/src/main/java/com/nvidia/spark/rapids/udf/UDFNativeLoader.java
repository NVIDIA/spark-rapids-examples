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

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.URL;

/** Loads the native dependencies for UDFs with a native implementation */
public class UDFNativeLoader {
  private static final ClassLoader loader = UDFNativeLoader.class.getClassLoader();
  private static boolean isLoaded;

  /** Loads native UDF code if necessary */
  public static synchronized void ensureLoaded() {
    if (!isLoaded) {
      try {
        String os = System.getProperty("os.name");
        String arch = System.getProperty("os.arch");
        File path = createFile(os, arch, "spatialudfjni");
        System.load(path.getAbsolutePath());
        isLoaded = true;
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
    }
  }

  /** Extract the contents of a library resource into a temporary file */
  private static File createFile(String os, String arch, String baseName) throws IOException {
    String path = arch + "/" + os + "/" + System.mapLibraryName(baseName);
    File loc;
    URL resource = loader.getResource(path);
    if (resource == null) {
      throw new FileNotFoundException("Could not locate native dependency " + path);
    }
    try (InputStream in = resource.openStream()) {
      loc = File.createTempFile(baseName, ".so");
      loc.deleteOnExit();
      try (OutputStream out = new FileOutputStream(loc)) {
        byte[] buffer = new byte[1024 * 16];
        int read = 0;
        while ((read = in.read(buffer)) >= 0) {
          out.write(buffer, 0, read);
        }
      }
    }
    return loc;
  }
}
