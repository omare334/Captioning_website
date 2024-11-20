from minio import Minio
from minio.error import S3Error
import os
from pathlib import Path
from time import sleep

# MinIO Configuration
MINIO_URL = "minio:9000"  # Use 'minio' as it is the Docker service name
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin123"


def init_minio():
        max_retries = 5
        retry_delay = 3


        for attempt in range(max_retries):
            try:
                # Initialize MinIO client
                client = Minio(
                    MINIO_URL,
                    access_key=MINIO_ACCESS_KEY,
                    secret_key=MINIO_SECRET_KEY,
                    secure=os.getenv("MINIO_SECURE", "false").lower() == "true", 
                )
                
                # Test connection
                client.bucket_exists("test")
                break  # If successful, break the retry loop
                
            except Exception as e:
                if attempt == max_retries - 1:  # Last attempt
                    raise Exception(f"Failed to connect to MinIO after {max_retries} attempts: {e}")
                print(f"Failed to connect to MinIO, retrying in {retry_delay} seconds...")
                sleep(retry_delay)

        bucket_name = "data-models"  # Changed from data_models to data-models
        try:
            if not client.bucket_exists(bucket_name):
                client.make_bucket(bucket_name)
                print(f"Created bucket: {bucket_name}")
            else:
                print(f"Bucket {bucket_name} already exists")
        except Exception as e:
            print(f"Error with bucket creation: {e}")

        # Uploading a file to MinIO
        files_to_upload = [
                "word-vector-embeddings.model",
                "training-with-tokens.parquet",
                "doc-index-64.faiss",
                "two_tower_state_dict.pth",
            ]

        # Upload files from /data directory only if they don't exist in the bucket
        for file in files_to_upload:
            source_path = Path("/data-models") / file
            try:
                # Check if file exists in bucket
                client.stat_object("data-models", file)
                print(f"File {file} already exists in bucket")
            except:
                if source_path.exists():
                    try:
                        print(f"Uploading {file} to bucket")
                        client.fput_object("data-models", file, str(source_path))
                        print(f"Successfully uploaded {file}")
                    except Exception as e:
                        print(f"Error uploading {file}: {e}")
                else:
                    print(f"Warning: Source file not found: {source_path}")

if __name__ == "__main__":
    init_minio()