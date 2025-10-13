# GPU-Accelerated Spark Connect for ETL and ML (Spark 4.0)

This project demonstrates a complete GPU-accelerated ETL and Machine Learning pipeline using Apache Spark 4.0 with Spark Connect, featuring the RAPIDS Accelerator. The example showcases the capabilities presented in the Data and AI Summit 2025 session:
[GPU Accelerated Spark Connect](https://www.databricks.com/dataaisummit/session/gpu-accelerated-spark-connect).
It is similar to the XGBoost example in this repo.
The key difference is that it uses Spark Connect thus the notebook server node has no heavy dependencies and it uses
LogisticRegression to demonstrate accelerated Spark MLlib functionality

## 🚀 Key Features

- **Apache Spark 4.0** with cutting-edge Spark Connect capabilities
- **GPU acceleration** via RAPIDS Accelerator
- **MLlib over Spark Connect** - new in Spark 4.0
- **Zero-code-change acceleration** - existing Spark applications automatically benefit
- **Complete ETL and ML pipeline** demonstration with mortgage data
- **Jupyter Lab integration** for interactive development
- **Docker Compose** setup for easy deployment with clear distinction what dependencies are
required by what service and where GPUs are really used

## 📊 Performance Highlights

The included demonstration shows:
- **Comprehensive ETL pipeline** processing mortgage data with complex transformations for feature engineering
- **Machine Learning workflow** using Logistic Regression with Feature Hashing
- **GPU vs CPU performance comparison** with visualization of the speedup achieved on the hardware the demo is run

## 🏗️ Architecture

The setup consists of four Docker services:

1. **Spark Master** (`spark-master`) - Cluster coordination and job scheduling
2. **Spark Worker** (`spark-worker`) - GPU-enabled worker node for task execution. 
3. **Spark Connect Server** (`spark-connect-server`) - gRPC interface with RAPIDS integration
4. **Jupyter Lab Client** (`spark-connect-client`) - Interactive development environment

## 📋 Prerequisites

### Required
- [Docker](https://docs.docker.com/engine/install/) and [Docker Compose](https://docs.docker.com/compose/install/linux)
- At least 8GB of available RAM
- Available ports: 8080, 8081, 8888, 7077, 4040, 15002

### For GPU Acceleration
- NVIDIA GPU with CUDA compute capability supported by RAPIDS
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- Docker Compose version should be `2.30.x` or newer to avoid an NVIDIA Container Toolkit related bug.  [Update](https://docs.docker.com/compose/install/linux) if necessary
- CUDA 12.x drivers

## 🚀 Quick Start

1. **Clone and navigate to the project:**
   ```bash
   cd examples/spark-connect-for-etl-and-ml
   ```

2. **Set up data directory (if needed):**
   ```bash
   export WORK_DIR=$(pwd)/work
   export DATA_DIR=$(pwd)/data/mortgage.input.csv
   mkdir -p $WORK_DIR $DATA_DIR
   chmod 777 $WORK_DIR
   ```
   Download a few quarters worth of the [Mortgage Dataset](https://capitalmarkets.fanniemae.com/credit-risk-transfer/single-family-credit-risk-transfer/fannie-mae-single-family-loan-performance-data)
   to the `$DATA_DIR` location.

3. **Start all services:**
   ```bash
   docker-compose up -d
   ```
   (`docker compose` can be used in place of `docker-compose` here and throughout)

4. **Access the interfaces:**
   - **Jupyter Lab**: http://localhost:8888 (no password required)
   - **Spark Master UI**: http://localhost:8080
   - **Spark Worker UI**: access from Master UI link
   - **Spark Driver UI**: access from Master UI link (`Name` column of `Running Applications`)
   - If demo containers are run on a headless host, to access above URLs:
     - First, set up a port-forwarding ssh tunnel from your local host/laptop to the host running docker compose:
       ```bash
       ssh <user@docker-compose-host> -L 8888:localhost:8888 -L 8080:localhost:8080 -L 8081:localhost:8081 -L 4040:localhost:4040
       ```
     - Second, to avoid some broken UI links from the local web browser, add the line `127.0.0.1 spark-master` to your local `/etc/hosts` file (note, modifying the `hosts` file may need local sudo access)

5. **Open the demo notebook:**
   - Navigate to `work/spark-connect-demo.ipynb` in Jupyter Lab
   - You can also open it in VS Code by selecting http://localhost:8888 as the
     existing notebook server connection
   - Run the complete ETL and ML pipeline demonstration


## 📝 Demo Notebook Overview

The `spark-connect-demo.ipynb` notebook demonstrates:

### ETL Pipeline
- **Data ingestion** from CSV with custom schema
- **Complex transformations** including date parsing and delinquency calculations
- **String-to-numeric encoding** for categorical features
- **Data joins and aggregations** with mortgage performance data

### Machine Learning Workflow
- **Feature engineering** with FeatureHasher and VectorAssembler
- **Logistic Regression** training for multi-class prediction
- **Model evaluation** with performance metrics
- **GPU vs CPU timing comparisons**

### Key Code Examples

**Connecting to Spark with GPU acceleration:**
```python
from pyspark.sql import SparkSession

spark = (
  SparkSession.builder
    .remote('sc://spark-connect-server')
    .appName('GPU-Accelerated-ETL-ML-Demo')
    .getOrCreate()
)
```

**GPU acceleration test:**
```python
spark.conf.set('spark.rapids.sql.enabled', True)
df = (
  spark.range(2 ** 35)
    .withColumn('mod10', col('id') % lit(10))
    .groupBy('mod10').agg(count('*'))
    .orderBy('mod10')
)
df.explain(mode='extended')  # Shows GPU operations in physical plan
```

**Machine Learning with GPU acceleration:**
```python
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler, FeatureHasher

spark.conf.set('spark.connect.ml.backend.classes', 'com.nvidia.rapids.ml.Plugin')

# Feature preparation
hasher = FeatureHasher(inputCols=categorical_cols, outputCol='hashed_categorical')
assembler = VectorAssembler().setInputCols(numerical_cols + ['hashed_categorical']).setOutputCol('features')

# Model training
logistic = LogisticRegression().setFeaturesCol('features').setLabelCol('delinquency_12')
pipeline = Pipeline().setStages([hasher, assembler, logistic])
model = pipeline.fit(training_data)
```

## 🐳 Service Details

### Spark Master
- **Image**: `apache/spark:4.0.0`
- **Ports**: 8080 (Web UI), 7077 (Master)
- **Role**: Cluster coordination and resource management

### Spark Worker (the only GPU node role)
- **Image**: Custom build based on `apache/spark:4.0.0`
- **GPU**: NVIDIA GPU support via Docker Compose deploy configuration
- **Ports**: 8081 (Web UI)
- **Features**: GPU resource discovery and task execution

### Spark Connect Server
- **Image**: Custom build based on `apache/spark:4.0.0` with Spark RAPIDS ETL and ML Plugins
- **RAPIDS Version**: 25.08.0 for CUDA 12
- **Ports**: 15002 (gRPC), 4040 (Driver UI)
- **Configuration**: Optimized for GPU acceleration with memory management

### JupyterLab - Spark Connect Client
- **Image**: Based on `jupyter/minimal-notebook:latest`
- **Environment**: Pre-configured with PySpark Connect Client
- **Ports**: 8888 (Jupyter Lab)
- **Volumes**: Notebooks and work directory mounted

## 📊 Performance Monitoring

You can use tools like nvtop or jupyterlab_nvdashboard running on the GPU host(s)


## 🧹 Cleanup

Stop and remove all services:
```bash
docker-compose down -v
```

Remove built images:
```bash
docker-compose down --rmi all -v
```

## Troubleshooting

Repeated executions of the notebook sometimes results in unexpected side effects
such as a `FileNotFoundException`. To mitigate restart the spark-connect-server
service

```bash
$ WORK_DIR=~/work DATA_DIR=~/dais-2025/data/mortgage/raw docker compose restart spark-connect-server
```

and/or restart the Jupyter kernel

### Logs
Logs for the spark driver/connect server, standalone master, standalone worker, and jupyter server can be viewed using the respective commands:
```bash
docker logs spark-connect-server
docker logs spark-master
docker logs spark-worker
docker logs spark-connect-client
```

Spark executor logs can be accessed via the Spark UI as usual.

## 📖 Additional Resources

- [Apache Spark 4.0 Documentation](https://spark.apache.org/docs/latest/)
- [Spark Connect Guide](https://spark.apache.org/docs/latest/spark-connect-overview.html)
- [NVIDIA RAPIDS Accelerator](https://nvidia.github.io/spark-rapids/)
- [Data and AI Summit Session](https://www.databricks.com/dataaisummit/session/gpu-accelerated-spark-connect)
