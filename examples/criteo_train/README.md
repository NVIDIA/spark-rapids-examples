# Criteo

ETL and deep learning training of the Criteo 1TB Click Logs dataset. Users can prepare their dataset accordingly.

## Dataset

We are using the preprocessed data from DLRM. All 40 columns(1 label + 39 features) are already numeric.

In the following parts, we assume that the data are mounted as docker volumn at `/data/parquet`

## Run Criteo example benchmark using Dockerfile

1. Build the docker image
```
nvidia-docker build -f Dockerfile -t nvspark/tf-hvd-train:0.1 .
```

2. Enter into it (also mount necessary dataset volume and devices)
```
 nvidia-docker run \
 --network host \
 --device /dev/infiniband \
 --privileged \
 -v /raid/spark-team/criteo/parquet:/data/parquet \
 -it nvspark/tf-hvd-train-pure:0.1 bash
```

3. when you are inside the container
```
cd /workspace
# start standalone Spark
./start-spark.sh

# start training
./submit.sh
```

## Note: 

if you only want to try in a node with only 1 GPU, please modify the GPU number per worker in `$SPARK_HOME/conf/spark-env.sh` before you launch spark workers becasue the docker image is targeted for DGX-2 with 16 GPUs

## file decription:

`Dockerfile`: consistent environment, main components are build from source directly. But this file take a while to build an image.

`spark-env.sh`: Spark config changes, we set 16 GPU for a work for DGX-2 box. It is in SPARK_HOME/conf/

`start-spark.sh`: launch Spark in standalone mode. In a DGX-2 box, it will launch Spark Master and a Spark Worker which contains 16 GPUs

`submit.sh`: commands used to submit the job

`criteo_keras.py`: Python script to train the Criteo model. Please run `python criteo_keras.py --help` to see parameter details

## workspace folder in docker container:

`/workspace/`