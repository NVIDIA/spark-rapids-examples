
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
    
