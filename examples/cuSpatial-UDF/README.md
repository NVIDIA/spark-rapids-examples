# cuSpatialUDF

Examples of Rapids UDF leveraging cuSpatial

## Build
### Prerequisites:
1. essential build tools:
    - [cmake(>=3.20)](https://cmake.org/download/),
    - [ninja(>=1.10)](https://github.com/ninja-build/ninja/releases),
    - [gcc(>=9.3)](https://gcc.gnu.org/releases.html)
2. [CUDA Toolkit(>=11.0)](https://developer.nvidia.com/cuda-toolkit)
3. conda: use [miniconda](https://docs.conda.io/en/latest/miniconda.html) to maintain header files and cmake dependecies
4. [cuspatial](https://github.com/rapidsai/cuspatial): install libcuspatial
    ```Bash
    # get libcuspatial from conda
    conda install -c rapidsai -c nvidia -c conda-forge  -c defaults libcuspatial=22.02
    # or below command for the nightly (aka SNAPSHOT) version.
    conda install -c rapidsai-nightly -c nvidia -c conda-forge  -c defaults libcuspatial=22.04
    ```
## Build in Docker
1. Build the docker image [Dockerfile](./Dockerfile), then run the container.
     ```Bash
     docker build -f Dockerfile . -t build-cuspatial-udf
     docker run -it build-cuspatial-udf bash
     ```
 2. Get the code, then run "mvn package".
     ```Bash
     git clone https://github.com/NVIDIA/spark-rapids-examples.git
     cd spark-rapids-examples/examples/cuSpatial-UDF/
     mvn package
     ```
 3. You'll get "cuspatial-udf-22.04-SNAPSHOT.jar" in the target folder.

## Run
### Run on Standalone
Besides `cudf` and `cuspatial`, the `gdal` library that is compatible with the installed `cuspatial` may also be needed.

Install it by running the command below.
```
conda install -c conda-forge libgdal=3.3.1
```
Set up a standalone cluster of Spark.

Besides, prepare the points data files and shape data file, then update the data paths accordingly in the script `gpu-run.sh`, and run this script.
 ```Bash
 ./gpu-run.sh
 ```
 
 ### Run on AWS Databricks
 1. Build the customized docker image [Dockerfile.awsdb](./Dockerfile.awsdb) and push to dockerhub so that it can be accessible by AWS Databricks.
     ```Bash
     # replace your dockerhub repo, your tag or any other repo AWS DB can access
     docker build -f Dockerfile.awsdb . -t <your-dockerhub-repo>:<your-tag>
     docker push <your-dockerhub-repo>:<your-tag>
     ```
 
 2. Follow the [Spark-rapids get-started document](https://github.com/NVIDIA/spark-rapids/blob/branch-22.04/docs/get-started/getting-started-databricks.md#start-a-databricks-cluster) to create a GPU cluster on AWS Databricks.
 Something different from the above document.
    * Databricks Runtime Version
  You should choose a Standard version of the Runtime version like "Runtime: 9.1 LTS(Scala 2.12, Spark 3.1.2)" and choose GPU instance type like "g4dn.xlarge". When I choose a ML version, it says "Support for Databricks container services requires runtime version 5.3+" and I can't click "Confirm" button.
    * Use your own Docker container
  Input "Docker Image URL" as "<your-dockerhub-repo>:<your-tag>"
    * For the other configurations, you can follow the get-started document.

 3. Copy the sample [cuspatial_data.tar.gz](../../datasets/cuspatial_data.tar.gz) or your data to DBFS by using Databricks CLI.
    ```Bash
    # extract the data
    tar zxf cuspatial_data.tar.gz
    databricks fs cp -r cuspatial_data/* dbfs:/data/cuspatial_data/
    databricks fs ls dbfs:/data/cuspatial_data/
    # it should have below 2 folders.
        points
        polygons
    ```
 4. Import the Library "cuspatial-udf-22.04-SNAPSHOT.jar" to the Databricks, then install it to your cluster.
 5. Import [cuspatial_sample.ipynb](./notebooks/cuspatial_sample.ipynb) to your workspace in Databricks. Attach to your cluster, then run it.
