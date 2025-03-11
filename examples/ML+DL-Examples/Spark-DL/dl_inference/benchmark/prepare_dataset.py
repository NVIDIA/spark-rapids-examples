import os
import tarfile
import pandas as pd
from pyspark.sql import SparkSession

def prepare_imagenet_parquet(size='50k', data_dir="spark-dl-datasets/imagenet-val"):
    """Prepare ImageNet validation set as parquet file with raw bytes."""

    size_map = {
        '1k': 1000,
        '5k': 5000,
        '10k': 10000,
        '50k': 50000
    }
    num_images = size_map.get(size, 50000)

    valdata_path = os.path.join(data_dir, 'ILSVRC2012_img_val.tar')
    if not os.path.exists(valdata_path):
        raise RuntimeError(
            "ImageNet validation data not found. Please download:\n"
            "ILSVRC2012_img_val.tar\n"
            f"And place it in {data_dir}"
        )
    
    images = []
    count = 0
    
    # Write raw compressed JPEG bytes to parquet
    with tarfile.open(valdata_path, 'r:') as tar:
        members = tar.getmembers()
        for _, member in enumerate(members):
            if count >= num_images:
                break
                
            if member.isfile() and member.name.endswith(('.JPEG', '.jpg', '.jpeg')):
                f = tar.extractfile(member)
                if f is not None:
                    raw_bytes = f.read()
                    images.append(raw_bytes)
                    count += 1
                    
                    if count % 100 == 0:
                        print(f"Processed {count} images")

    pdf = pd.DataFrame({
        'value': images
    })
    return pdf

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=str, default='50k', help='Dataset size (e.g., 1k, 5k, 10k, 50k)')
    args = parser.parse_args()

    pdf = prepare_imagenet_parquet(size=args.size)
    if not os.path.exists("spark-dl-datasets"):
        os.makedirs("spark-dl-datasets")

    pdf.to_parquet(f"spark-dl-datasets/imagenet_{args.size}.parquet")

    # Repartition and write to parquet
    spark = SparkSession.builder.appName("prepare-imagenet-parquet").getOrCreate()

    spark.conf.set("spark.sql.execution.arrow.useLargeVarTypes", "true")
    spark.conf.set("spark.sql.parquet.columnarReaderBatchSize", "1024")
    spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "1024")

    df = spark.read.parquet(f"spark-dl-datasets/imagenet_{args.size}.parquet")
    df = df.repartition(16)
    df.write.mode("overwrite").parquet(f"spark-dl-datasets/imagenet_{args.size}.parquet")

if __name__ == "__main__":
    main()