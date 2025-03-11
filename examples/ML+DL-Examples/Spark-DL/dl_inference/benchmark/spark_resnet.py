import os
import pandas as pd
import numpy as np
import time
from typing import Iterator
from pyspark.sql.types import ArrayType, FloatType
from pyspark import TaskContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf, col
from pyspark.ml.functions import predict_batch_udf
from gpu_monitor import GPUMonitor

def predict_batch_fn():
    """Classify batch of images"""
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
    spark = SparkSession.builder.appName("bench-spark-resnet").getOrCreate()

    # Avoid OOM for image loading from raw byte arrays
    spark.conf.set("spark.sql.execution.arrow.useLargeVarTypes", "true")
    spark.conf.set("spark.sql.parquet.columnarReaderBatchSize", "1024")
    spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "1024")

    file_path = os.path.abspath("spark-dl-datasets/imagenet_10k.parquet")
    classify = predict_batch_udf(predict_batch_fn,
                                return_type=ArrayType(FloatType()),
                                input_tensor_shapes=[[3, 224, 224]],
                                batch_size=1024)
    
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

if __name__ == "__main__":
    main()
    