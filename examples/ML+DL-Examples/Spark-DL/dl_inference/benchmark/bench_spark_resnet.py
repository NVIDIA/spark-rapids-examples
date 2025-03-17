import os
import pandas as pd
import numpy as np
import time
import argparse
from functools import partial
from typing import Iterator
from pyspark.sql.types import ArrayType, FloatType
from pyspark import TaskContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf, col
from pyspark.ml.functions import predict_batch_udf
from gpu_monitor import GPUMonitor
from pytriton_utils import TritonServerManager

def triton_server(ports):
    """Initialize and run Triton server for inference"""
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

def predict_batch_fn():
    """Classify batch of images in-process"""
    import torch
    import torchvision.models as models

    start_load = time.perf_counter()
    model = models.resnet50(pretrained=True).to("cuda")
    model.eval()
    end_load = time.perf_counter()
    print(f"Model loaded in {end_load - start_load:.4f} seconds")

    def predict(inputs):
        print(f"PARTITION {TaskContext.get().partitionId()}: Inferring batch of size {len(inputs)}")
        batch_tensor = torch.from_numpy(inputs).to("cuda")
        
        with torch.no_grad():
            outputs = model(batch_tensor)

        _, predicted_ids = torch.max(outputs, 1)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidences = torch.max(probabilities, dim=1)[0]
        indices = predicted_ids.cpu().numpy()
        scores = confidences.cpu().numpy()
        results = np.stack([indices, scores], axis=1).astype(np.float32)
        return results
    
    return predict

def triton_fn(model_name, host_to_url):
    """Send/receive batch of images to/from Triton server for inference"""
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

@pandas_udf(ArrayType(FloatType()))
def preprocess(image_iter: Iterator[pd.Series]) -> Iterator[pd.Series]:
    """Preprocess images (raw JPEG bytes) into a batch of tensors"""
    import io
    from PIL import Image
    from torchvision import transforms
    import torch
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-size', type=str, default='50k', help='Dataset size (e.g., 1k, 5k, 10k, 50k)')
    parser.add_argument('--batch-size', type=int, default=1024, help='Batch size used in predict_batch_udf')
    parser.add_argument('--use-triton', action='store_true', help='Use Triton for inference')
    args = parser.parse_args()

    spark = SparkSession.builder.appName("bench-spark-resnet").getOrCreate()

    # Avoid OOM for image loading from raw byte arrays
    spark.conf.set("spark.sql.execution.arrow.useLargeVarTypes", "true")
    spark.conf.set("spark.sql.parquet.columnarReaderBatchSize", "1024")
    spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "1024")

    server_manager = None
    predict_fn = None
    if args.use_triton:
        # Start server if using Triton
        spark.sparkContext.addPyFile("pytriton_utils.py")
        model_name = "resnet50"
        server_manager = TritonServerManager(model_name)
        server_manager.start_servers(triton_server)
        host_to_grpc_url = server_manager.host_to_grpc_url
        predict_fn = partial(triton_fn, model_name=model_name, host_to_url=host_to_grpc_url)
    else:
        predict_fn = predict_batch_fn

    # Define predict_batch_udf
    classify = predict_batch_udf(
            predict_fn,
            return_type=ArrayType(FloatType()),
            input_tensor_shapes=[[3, 224, 224]],
            batch_size=args.batch_size
        )
    
    # Define file path
    file_path = os.path.abspath(f"spark-dl-datasets/imagenet_{args.dataset_size}.parquet")

    # Start GPU utilization monitoring
    monitor = GPUMonitor()
    monitor.start()

    try:
        start_read = time.perf_counter()

        # Read -> Preprocess -> Classify -> Write
        df = spark.read.parquet(file_path)
        preprocessed_df = df.withColumn("images", preprocess(col("value"))).drop("value")
        preds = preprocessed_df.withColumn("preds", classify(col("images")))
        preds.write.mode("overwrite").parquet(f"spark-dl-datasets/imagenet_{args.dataset_size}_preds.parquet")

        end_write = time.perf_counter()

        print(f"E2E read -> inference -> write time: {end_write - start_read:.4f} seconds")
    finally:
        monitor.stop()
        if server_manager:
            server_manager.stop_servers()

if __name__ == "__main__":
    main()
