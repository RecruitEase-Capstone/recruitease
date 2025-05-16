from minio import Minio
from io import BytesIO
import tempfile
import os

class MinioClient:
    def __init__(self, endpoint:str, access_key:str, secret_key:str, secure:bool=False):
        self.client = Minio(
            endpoint=endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure
        )

        # Kode untuk minio.py
    def download_file(self, bucket_name: str, object_name: str):
        """
        Download file dari MinIO dan langsung return path ke file temporary.
        """
        # Buat temporary file dengan suffix .pdf
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        temp_path = temp_file.name
        temp_file.close()
        
        # Download file dari minio ke temporary file
        self.client.fget_object(bucket_name, object_name, temp_path)
        
        # Return path ke file
        return temp_path