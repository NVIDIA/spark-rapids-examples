# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import cudf
import numpy as np
import cupy
import sys


def read_points(path):
    print('reading points file:', path)
    points = np.fromfile(path, dtype=np.int32)
    points = cupy.asarray(points)
    points = points.reshape((len(points)// 4, 4))
    points = cudf.DataFrame(points)
    points_df = cudf.DataFrame()
    points_df['x'] = points[0]
    points_df['y'] = points[1]
    return points_df

if __name__ == '__main__':
    if len(sys.argv) < 3:
        raise Exception("Usage: to_parquet <input data path> <output data path>.")
    inputPath = sys.argv[1]
    outputPath = sys.argv[2]

    points_df = read_points(inputPath)
    points_df.to_parquet(outputPath)
    
