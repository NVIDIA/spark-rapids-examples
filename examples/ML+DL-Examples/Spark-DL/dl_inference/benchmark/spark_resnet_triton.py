import os
import pandas as pd
import time
from functools import partial
from typing import Iterator
from pyspark.sql.types import ArrayType, FloatType
from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf, col
from pyspark.ml.functions import predict_batch_udf
from gpu_monitor import GPUMonitor
from pytriton_utils import TritonServerManager

def triton_server(ports):
    import time
    import signal
    import numpy as np
    from pytriton.decorators import batch
    from pytriton.model_config import DynamicBatcher, ModelConfig, Tensor
    from pytriton.triton import Triton, TritonConfig
    from pyspark import TaskContext
    import torch
    import torchvision.models as models

    print(f"SERVER: Initializing model on worker {TaskContext.get().partitionId()}.")
    model = models.resnet50(pretrained=True)
    model = model.to("cuda")
    model.eval()

    @batch
    def _infer_fn(**inputs):
        images = inputs["images"]
        print(f"SERVER {time.time()}: Received batch of size {len(images)}")
        batch_tensor = torch.from_numpy(images).to("cuda")

        with torch.no_grad():
            outputs = model(batch_tensor)
            
        _, predicted_ids = torch.max(outputs, 1)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidences = torch.max(probabilities, dim=1)[0]
        indices = predicted_ids.cpu().numpy()
        scores = confidences.cpu().numpy()
        results = np.stack([indices, scores], axis=1).astype(np.float32)
        return {
            "preds": results,
        }

    workspace_path = f"/tmp/triton_{TaskContext.get().partitionId()}_{time.strftime('%m_%d_%M_%S')}"
    triton_conf = TritonConfig(http_port=ports[0], grpc_port=ports[1], metrics_port=ports[2])
    with Triton(config=triton_conf, workspace=workspace_path) as triton:
        triton.bind(
            model_name="resnet50",
            infer_func=_infer_fn,
            inputs=[
                Tensor(name="images", dtype=np.float32, shape=(3, 224, 224)),
            ],
            outputs=[
                Tensor(name="preds", dtype=np.float32, shape=(2,)),
            ],
            config=ModelConfig(
                max_batch_size=1024,
                batcher=DynamicBatcher(max_queue_delay_microseconds=500000),  # 500ms
            ),
            strict=True,
        )

        def _stop_triton(signum, frame):
            print("SERVER: Received SIGTERM. Stopping Triton server.")
            triton.stop()

        signal.signal(signal.SIGTERM, _stop_triton)

        print("SERVER: Serving inference")
        triton.serve()

@pandas_udf(ArrayType(FloatType()))
def preprocess(image_iter: Iterator[pd.Series]) -> Iterator[pd.Series]:
    """Preprocess images (raw JPEG bytes) into a batch of tensors"""
    import torch
    from PIL import Image
    from torchvision import transforms
    import io
    from pyspark import TaskContext

    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    part_id = TaskContext.get().partitionId()

    for image_batch in image_iter:
        batch_size = len(image_batch)
        print(f"PARTITION {part_id}: number of images: {batch_size}")

        # Pre-allocate tensor for batch
        batch_tensor = torch.empty(batch_size, 3, 224, 224, dtype=torch.float32)
        
        # Decompress and transform images
        for idx, raw_bytes in enumerate(image_batch):
            img = Image.open(io.BytesIO(raw_bytes))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            batch_tensor[idx] = preprocess(img)

        numpy_batch = batch_tensor.numpy()
        flattened_batch = numpy_batch.reshape(batch_size, -1)
        
        yield pd.Series(list(flattened_batch))

def triton_fn(model_name, host_to_url):
    from pytriton.client import ModelClient
    import socket
    from pyspark import TaskContext

    url = host_to_url.get(socket.gethostname())
    client = ModelClient(url, model_name, inference_timeout_s=500)
    part_id = TaskContext.get().partitionId()
    print(f"PARTITION {part_id}: Connecting to Triton model {model_name} at {url}.")

    def infer_batch(inputs):
        print(f"PARTITION {part_id}: Inferring batch of size {len(inputs)}")
        result_data = client.infer_batch(inputs)
        result_data = result_data["preds"]
        return result_data
        
    return infer_batch

def main():
    spark = SparkSession.builder.appName("bench-spark-resnet-triton").getOrCreate()
    spark.sparkContext.addPyFile("pytriton_utils.py")

    # Avoid OOM for image loading from raw byte arrays
    spark.conf.set("spark.sql.execution.arrow.useLargeVarTypes", "true")
    spark.conf.set("spark.sql.parquet.columnarReaderBatchSize", "1024")
    spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "1024")

    # Start server
    model_name = "resnet50"
    num_nodes = 1
    server_manager = TritonServerManager(num_nodes, model_name)
    server_manager.start_servers(triton_server, wait_timeout=24)
    host_to_grpc_url = server_manager.host_to_grpc_url

    file_path = os.path.abspath("spark-dl-datasets/imagenet_10k.parquet")
    classify = predict_batch_udf(partial(triton_fn, model_name=model_name, host_to_url=host_to_grpc_url),
                                return_type=ArrayType(FloatType()),
                                input_tensor_shapes=[[3, 224, 224]],
                                batch_size=256)
    
    # Start GPU utilization monitoring
    monitor = GPUMonitor()
    monitor.start()

    try:
        start_read = time.perf_counter()

        df = spark.read.parquet(file_path)
        preprocessed_df = df.withColumn("images", preprocess(col("value"))).drop("value")
        preds = preprocessed_df.withColumn("preds", classify(col("images")))
        preds.write.mode("overwrite").parquet(f"spark-dl-datasets/imagenet_10k_preds.parquet")

        end_write = time.perf_counter()

        print(f"E2E read -> inference -> write time: {end_write - start_read:.4f} seconds")
    finally:
        monitor.stop()
        server_manager.stop_servers()

if __name__ == "__main__":
    main()