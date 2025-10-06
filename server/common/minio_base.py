"""Base MinIO client class for shared functionality."""

import os
import io
from typing import Optional, BinaryIO
import uuid

# Optional imports - allow module to load even if minio is not installed
try:
    from minio import Minio
    from minio.error import S3Error
    MINIO_AVAILABLE = True
except ImportError:
    Minio = None
    S3Error = Exception  # Fallback
    MINIO_AVAILABLE = False


class BaseMinIOClient:
    """Base class for MinIO clients with common functionality."""

    def __init__(self, bucket_name: str):
        self.endpoint = os.getenv("MINIO_ENDPOINT", "minio:9000")
        self.access_key = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
        self.secret_key = os.getenv("MINIO_SECRET_KEY", "minioadmin")
        self.bucket_name = bucket_name
        self._client = None
        self._available = False

    @property
    def client(self):
        if self._client is None:
            if not MINIO_AVAILABLE:
                print("MinIO package not installed. Using local storage only.")
                self._available = False
                return None
            try:
                self._client = Minio(
                    self.endpoint,
                    access_key=self.access_key,
                    secret_key=self.secret_key,
                    secure=False  # For local development
                )
                self._ensure_bucket()
                self._available = True
            except Exception as e:
                print(f"MinIO not available: {e}. Using local storage only.")
                self._available = False
                return None
        return self._client

    def _ensure_bucket(self):
        """Create bucket if it doesn't exist"""
        try:
            if not self._client.bucket_exists(self.bucket_name):
                self._client.make_bucket(self.bucket_name)
        except S3Error as e:
            print(f"Error creating bucket: {e}")

    def upload_content(self, content: str, object_id: str, content_type: str = 'text/plain') -> str:
        """Upload content to MinIO and return the key"""
        if not self._available or self.client is None:
            # MinIO not available, return empty key (content stored in DB)
            return ""

        try:
            # Create a unique key
            key = f"{self.bucket_name}/{object_id}/{uuid.uuid4()}.txt"

            # Convert string to bytes
            content_bytes = content.encode('utf-8')
            content_stream = io.BytesIO(content_bytes)

            # Upload to MinIO
            self.client.put_object(
                bucket_name=self.bucket_name,
                object_name=key,
                data=content_stream,
                length=len(content_bytes),
                content_type=content_type
            )

            return key
        except S3Error as e:
            raise Exception(f"Failed to upload content to MinIO: {e}")

    def download_content(self, key: str) -> str:
        """Download content from MinIO"""
        if not self._available or self.client is None or not key:
            return ""  # Return empty if MinIO not available or no key

        try:
            response = self.client.get_object(self.bucket_name, key)
            content = response.read().decode('utf-8')
            response.close()
            response.release_conn()
            return content
        except S3Error as e:
            raise Exception(f"Failed to download content from MinIO: {e}")

    def delete_content(self, key: str):
        """Delete content from MinIO"""
        if not self._available or self.client is None or not key:
            return  # Skip if MinIO not available or no key

        try:
            self.client.remove_object(self.bucket_name, key)
        except S3Error as e:
            raise Exception(f"Failed to delete content from MinIO: {e}")