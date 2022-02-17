# cuSpatialUDF

Examples of Rapids UDF leveraging cuSpatial

## Build
Build requires to install libraries of `cuspatial` and `cudf`. This can be done via conda by running the command below.
```Bash
conda install -c rapidsai -c nvidia -c conda-forge  -c defaults libcuspatial=22.02
```
or
```Bash
conda install -c rapidsai-nightly -c nvidia -c conda-forge  -c defaults libcuspatial=22.04
```
for the nightly (aka SNAPSHOT) version.

## Run
Besides `cudf` and `cuspatial`, the `gdal` library that is compatible with the installed `cuspatial` may also be needed.

Install it by running the command below.
```
conda install -c conda-forge libgdal
```
Set up a standalone cluster of Spark.

Besides, prepare the points data files and shape data file, then update the data paths accordingly in the script `gpu-run.sh`, and run this script.
 ```Bash
 ./gpu-run.sh
 ```