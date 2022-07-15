# Spark-cuSpatial

This is a Spark RapidsUDF application to illustrate how to use [cuSpatial](https://github.com/rapidsai/cuspatial) to solve a point-in-polygon problem.
It implements a [RapidsUDF](https://nvidia.github.io/spark-rapids/docs/additional-functionality/rapids-udfs.html#adding-gpu-implementations-to-udfs) 
interface to call the cuSpatial functions through JNI. It can be run on a distributed Spark cluster with scalability.

## Performance
We got the end-2-end hot run times as below table when running with 2009 NYC Taxi trip pickup location,
which includes 170,896,055 points, and 3 sets of polygons(taxi_zone, nyct2000, nycd Community-Districts).
The point data can be downloaded from [TLC Trip Record Data](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page).
The polygon data can be downloaded from [taxi_zone dataset](https://data.cityofnewyork.us/Transportation/NYC-Taxi-Zones/d3c5-ddgc),
[nyct2000 dataset](https://data.cityofnewyork.us/City-Government/2000-Census-Tracts/ysjj-vb9j) and 
[nycd Community-Districts dataset](https://data.cityofnewyork.us/City-Government/Community-Districts/yfnk-k7r4)

| Environment | Taxi_zones (263 Polygons) | Nyct2000 (2216 Polygons) | Nycd Community-Districts (71 Complex Polygons)|
| ----------- | :---------: | :---------: | :---------: |
| 4-core CPU | 3.9 minutes | 4.0 minutes| 4.1 minutes |
| 1 GPU(T4) on Databricks | 25 seconds | 27 seconds | 28 seconds|
| 2 GPU(T4) on Databricks | 15 seconds | 14 seconds | 17 seconds |
| 4 GPU(T4) on Databricks | 11 seconds | 11 seconds | 12 seconds |

Note: Please update the `x,y` column names to `Start_Lon,Start_Lat` in
the [notebook](./notebooks/cuspatial_sample_db.ipynb) if you test with the download points.

taxi-zones map:

<img src="../../../docs/img/guides/cuspatial/taxi-zones.png" width="600">

nyct2000 map:

<img src="../../../docs/img/guides/cuspatial/Nyct2000.png" width="600">

nyct-community-districts map:

<img src="../../../docs/img/guides/cuspatial/Nycd-Community-Districts.png" width="600">

## Build
Firstly build the UDF JAR from source code before running this demo.
You can build the JAR [in Docker](#build-in-docker) with the provided [Dockerfile](Dockerfile), 
or [in local machine](#build-in-local-machine) after prerequisites.

### Build in Docker
1. Build the docker image [Dockerfile](Dockerfile), then run the container.
     ```Bash
     docker build -f Dockerfile . -t build-spark-cuspatial
     docker run -it build-spark-cuspatial bash
     ```
2. Get the code, then run `mvn package`.
     ```Bash
     git clone https://github.com/NVIDIA/spark-rapids-examples.git
     cd spark-rapids-examples/examples/UDF-Examples/Spark-cuSpatial/
     mvn package
     ```
3. You'll get the jar named `spark-cuspatial-<version>.jar` in the target folder.

Note: The docker env is just for building the jar, not for running the application.

### Build in Local:
1. Essential build tools:
    - [cmake(>=3.20)](https://cmake.org/download/),
    - [ninja(>=1.10)](https://github.com/ninja-build/ninja/releases),
    - [gcc(>=9.3)](https://gcc.gnu.org/releases.html)
2. [CUDA Toolkit(>=11.0)](https://developer.nvidia.com/cuda-toolkit)
3. conda: use [miniconda](https://docs.conda.io/en/latest/miniconda.html) to maintain header files and cmake dependecies
4. [cuspatial](https://github.com/rapidsai/cuspatial): install libcuspatial
    ```Bash
    # get libcuspatial from conda
    conda install -c rapidsai -c nvidia -c conda-forge  -c defaults libcuspatial=22.06
    # or below command for the nightly (aka SNAPSHOT) version.
    conda install -c rapidsai-nightly -c nvidia -c conda-forge  -c defaults libcuspatial=22.08
    ```
5. Get the code, then run `mvn package`.
     ```Bash
     git clone https://github.com/NVIDIA/spark-rapids-examples.git
     cd spark-rapids-examples/examples/Spark-cuSpatial/
     mvn package
     ```
6. You'll get `spark-cuspatial-<version>.jar` in the target folder.

## Run
### GPU Demo on Spark Standalone on-premises cluster
1. Install necessary libraries. Besides `cudf` and `cuspatial`, the `gdal` library that is compatible with the installed `cuspatial` may also be needed.
    Install it by running the command below.
    ```
    conda install -c conda-forge libgdal=3.3.1
    ```
2. Set up [a standalone cluster](/docs/get-started/xgboost-examples/on-prem-cluster/standalone-scala.md) of Spark. Make sure the conda/lib is included in LD_LIBRARY_PATH, so that spark executors can load libcuspatial.so.

3. Download spark-rapids jars
   * [spark-rapids v22.06.0](https://repo1.maven.org/maven2/com/nvidia/rapids-4-spark_2.12/22.06.0/rapids-4-spark_2.12-22.06.0.jar) or above
4. Prepare the dataset & jars. Copy the sample dataset from [cuspatial_data](../../../datasets/cuspatial_data.tar.gz) to `/data/cuspatial_data`.
    Copy spark-rapids & `spark-cuspatial-<version>.jar` to `/data/cuspatial_data/jars`.
    If you build the `spark-cuspatial-22.08.0-SNAPSHOT.jar` in docker, please copy the jar from docker to local:
    ```
    docker cp your-instance:/root/spark-rapids-examples/examples/UDF-Examples/Spark-cuSpatial/target/spark-cuspatial-22.08.0-SNAPSHOT.jar ./your-local-path
    ```
    You can use your own path, but remember to update the paths in `gpu-run.sh` accordingly.
5. Run `gpu-run.sh`
    ```Bash
    ./gpu-run.sh
    ```
### GPU Demo on AWS Databricks
1. Build a customized docker image using [Dockerfile.awsdb](Dockerfile.awsdb) and push to a Docker registry such as [Docker Hub](https://hub.docker.com/) which can be accessible by AWS Databricks.
     ```Bash
     # replace your dockerhub repo, your tag or any other repo AWS DB can access
     docker build -f Dockerfile.awsdb . -t <your-dockerhub-repo>:<your-tag>
     docker push <your-dockerhub-repo>:<your-tag>
     ```
 
2. Follow the [Spark-rapids get-started document](https://nvidia.github.io/spark-rapids/docs/get-started/getting-started-databricks.html#start-a-databricks-cluster) to create a GPU cluster on AWS Databricks.
 Something different from the document.
    * Databricks Runtime Version
  You should choose a Standard version of the Runtime version like `Runtime: 9.1 LTS(Scala 2.12, Spark 3.1.2)` and
  choose GPU instance type like `g4dn.xlarge`. Note that ML runtime does not support customized docker container.
`Support for Databricks container services requires runtime version 5.3+` 
  and the `Confirm` button is disabled.
    * Use your own Docker container
  Input `Docker Image URL` as `your-dockerhub-repo:your-tag`
    * Follow the Databricks get-started document for other steps.

3. Copy the sample [cuspatial_data.tar.gz](../../../datasets/cuspatial_data.tar.gz) or your data to DBFS by using Databricks CLI.
    ```Bash
    # extract the data
    tar zxf cuspatial_data.tar.gz
    databricks fs cp -r cuspatial_data/* dbfs:/data/cuspatial_data/
    databricks fs ls dbfs:/data/cuspatial_data/
    # it should have below 2 folders.
        points
        polygons
    ```
   The sample points and polygons are randomly generated.
   
   Sample polygons: 

   <img src="../../../docs/img/guides/cuspatial/sample-polygon.png" width="600">
   
4. Upload `spark-cuspatial-<version>.jar` on dbfs and then install it in Databricks cluster.
   
   <img src="../../../docs/img/guides/cuspatial/install-jar.png" width="600">    

5. Import [cuspatial_sample.ipynb](notebooks/cuspatial_sample_db.ipynb) to Databricks workspace, attach it to Databricks cluster and run it.

### CPU Demo on AWS Databricks
1. Create a Databricks cluster. For example, Databricks Runtime 10.3.

2. Install the Sedona jars and Sedona Python libs on Databricks using web UI. 
   The Sedona version should be 1.1.1-incubating or higher.
   * Install below jars from Maven Coordinates in Libraries tab:
    ```Bash
    org.apache.sedona:sedona-python-adapter-3.0_2.12:1.2.0-incubating
    org.datasyslab:geotools-wrapper:1.1.0-25.2
    ```
   * To enable python support, install below python lib from PyPI in Libraries tab 
    ```Bash
    apache-sedona
    ```
3. From your cluster configuration (Cluster -> Edit -> Configuration -> Advanced options -> Spark) activate the 
   Sedona functions and the kryo serializer by adding below to the Spark Config
    ```Bash
    spark.sql.extensions org.apache.sedona.viz.sql.SedonaVizExtensions,org.apache.sedona.sql.SedonaSqlExtensions
    spark.serializer org.apache.spark.serializer.KryoSerializer
    spark.kryo.registrator org.apache.sedona.core.serde.SedonaKryoRegistrator
    ```
   
4. Upload the sample data files to DBFS, start the cluster, attach the [notebook](notebooks/spacial-cpu-apache-sedona_db.ipynb) to the cluster, and run it.
