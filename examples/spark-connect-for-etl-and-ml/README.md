# GPU-Accelerated Spark Connect for ETL and ML (Spark 4.0)

This project provides a complete Docker Compose setup for Apache Spark 4.0 with GPU acceleration, showcasing the capabilities demonstrated in the [Data and AI Summit 2025 session: "GPU Accelerated Spark Connect"](https://www.databricks.com/dataaisummit/session/gpu-accelerated-spark-connect).

## Key Features

- **Apache Spark 4.0** with latest Spark Connect capabilities
- **MLlib over Spark Connect** (new in Spark 4.0)
- **NVIDIA RAPIDS GPU acceleration** for up to 9x performance improvement at 80% cost reduction
- **No-code change GPU acceleration** via plugin interface
- **End-to-end ETL and ML workflows** with Jupyter Lab integration

## Architecture

The setup consists of three main services with GPU acceleration:

1. **Spark Standalone Cluster** - GPU-enabled Master and Worker nodes for distributed processing
2. **Spark Connect Server** - gRPC-based interface with RAPIDS GPU plugin integration
3. **Jupyter Lab** - Interactive development environment with PySpark 4.0 and ML libraries

## Prerequisites

### Required
- Docker and Docker Compose installed
- At least 6GB of available RAM (increased for GPU workloads)
- Ports 8080, 8081, 8888, 7077, and 15002 available

### For GPU Acceleration (Optional)
- NVIDIA GPU with CUDA support (compute capability 6.0+)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed
- CUDA 11.x or 12.x drivers

*Note: The setup works without GPU - operations will fall back to CPU execution*

## Quick Start

1. **Start the services:**
   ```bash
   docker-compose up -d
   ```

2. **Access the interfaces:**
   - **Jupyter Lab**: http://localhost:8888 (no password required)
   - **Spark Master UI**: http://localhost:8080
   - **Spark Worker UI**: http://localhost:8081

3. **Connect to GPU-accelerated Spark from Jupyter:**
   ```python
   from pyspark.sql import SparkSession
   
   # Using Spark Connect 4.0 with GPU acceleration
   spark = SparkSession.builder \
       .remote("sc://localhost:15002") \
       .appName("GPU-ETL-ML-Example") \
       .getOrCreate()
   
   # Test GPU-accelerated operations
   from pyspark.sql.functions import *
   df = spark.range(100000).toDF("number")
   df = df.withColumn("squared", col("number") ** 2)
   df.show()
   
   # Check for GPU acceleration
   spark.sql("SHOW FUNCTIONS").filter(col("function").contains("gpu")).show()
   ```

## Service Details

### Spark Master & Worker (GPU-enabled)
- **Master**: Coordinates the cluster and schedules jobs with GPU awareness
- **Worker**: Executes tasks with 2 cores, 4GB memory, and GPU resource management
- **GPU Support**: NVIDIA Container Runtime integration for GPU task scheduling
- **UI Access**: Monitor jobs, GPU utilization, and cluster status via web interfaces

### Spark Connect Server (GPU-accelerated)
- **Port**: 15002 (gRPC)
- **Protocol**: Arrow-optimized for fast data transfer
- **GPU Plugin**: RAPIDS Accelerator integrated for transparent acceleration
- **Features**: 
  - Remote Spark session management
  - GPU-accelerated DataFrame operations
  - MLlib over Spark Connect (Spark 4.0 feature)

### Jupyter Lab (Spark 4.0)
- **Environment**: Pre-configured with PySpark 4.0 and ML libraries
- **Features**: Full MLlib support over Spark Connect
- **Volumes**: 
  - `./notebooks` → `/home/jovyan/work` (your notebooks)
  - `./data` → `/home/jovyan/data` (datasets)

## GPU Acceleration Details

This setup implements the architecture described in the [Data and AI Summit session](https://www.databricks.com/dataaisummit/session/gpu-accelerated-spark-connect), providing:

- **Transparent GPU acceleration** via RAPIDS plugin interface
- **No code changes required** - existing Spark applications automatically accelerated
- **Up to 9x performance improvement** on supported operations
- **80% cost reduction** through faster execution times
- **Automatic fallback** to CPU when GPU operations aren't supported

## Directory Structure

```
spark-connect-for-etl-and-ml/
├── docker-compose.yaml      # Main orchestration file
├── spark-defaults.conf      # Spark configuration
├── requirements.txt         # Python dependencies
├── data/                   # Shared data directory
├── notebooks/              # Jupyter notebooks
└── README.md               # This file
```

## Example Workflows

### ETL Pipeline Example

```python
# In Jupyter Lab
from pyspark.sql import SparkSession
from pyspark.sql.functions import *

# Connect to Spark
spark = SparkSession.builder.remote("sc://localhost:15002").getOrCreate()

# Read data
df = spark.read.parquet("/home/jovyan/data/input.parquet")

# Transform
transformed_df = df \
    .filter(col("status") == "active") \
    .withColumn("processed_date", current_timestamp()) \
    .groupBy("category") \
    .agg(count("*").alias("count"))

# Write results
transformed_df.write \
    .mode("overwrite") \
    .parquet("/home/jovyan/data/output.parquet")
```

### ML Workflow Example

```python
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline

# Prepare features
assembler = VectorAssembler(
    inputCols=["feature1", "feature2", "feature3"],
    outputCol="features"
)

# Create model
lr = LinearRegression(featuresCol="features", labelCol="target")

# Build pipeline
pipeline = Pipeline(stages=[assembler, lr])

# Train model
model = pipeline.fit(training_df)

# Make predictions
predictions = model.transform(test_df)
```

## Configuration

### Spark Configuration
Edit `spark-defaults.conf` to customize:
- Memory allocation
- Connect server settings
- Performance tuning
- Logging levels

### Python Dependencies
Modify `requirements.txt` to add additional packages:
```bash
# Add to requirements.txt
tensorflow>=2.13.0
torch>=2.0.0
```

Then rebuild:
```bash
docker-compose down
docker-compose up --build -d
```

## Data Management

### Adding Data
Place your datasets in the `data/` directory:
```bash
cp your-dataset.csv data/
cp your-parquet-files.parquet data/
```

### Accessing Data in Notebooks
```python
# Read CSV
df = spark.read.option("header", "true").csv("/home/jovyan/data/your-dataset.csv")

# Read Parquet
df = spark.read.parquet("/home/jovyan/data/your-parquet-files.parquet")
```

## Monitoring and Debugging

### Check Service Health
```bash
# View logs
docker-compose logs spark-master
docker-compose logs spark-connect
docker-compose logs jupyter

# Check service status
docker-compose ps
```

### Performance Monitoring
- **Spark UI**: http://localhost:8080 - Monitor jobs, stages, and executors
- **Worker UI**: http://localhost:8081 - View worker resource usage

### Common Issues

1. **Port conflicts**: Ensure ports 8080, 8081, 8888, 7077, 15002 are available
2. **Memory issues**: Increase Docker memory allocation or reduce Spark worker memory
3. **Connection timeouts**: Check network connectivity between containers

## Scaling

### Add More Workers
Modify `docker-compose.yaml` to add additional worker services:
```yaml
spark-worker-2:
  # Copy spark-worker configuration
  # Change container_name and ports
```

### Increase Resources
Adjust memory and CPU allocation in the compose file:
```yaml
environment:
  - SPARK_WORKER_MEMORY=4G
  - SPARK_WORKER_CORES=4
```

## Cleanup

Stop and remove all services:
```bash
docker-compose down -v
```

## Advanced Usage

### Custom Spark Applications
Submit standalone applications:
```bash
docker exec spark-master /opt/spark/bin/spark-submit \
  --master spark://spark-master:7077 \
  --class MySparkApp \
  /path/to/your-app.jar
```

### Integration with External Systems
- Configure database connections in `spark-defaults.conf`
- Mount credential files via Docker volumes
- Use environment variables for sensitive configuration

## Support

For issues and questions:
1. Check the logs: `docker-compose logs [service-name]`
2. Verify configuration files
3. Ensure adequate system resources
4. Review Spark documentation for advanced configuration

