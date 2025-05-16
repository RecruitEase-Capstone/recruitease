import grpc
import os
import logging

from src.proto import cv_processor_pb2_grpc as pb2_grpc
from concurrent import futures
from service.summarize_service import SummarizerService
from src.helper.minio.minio import MinioClient
from dotenv import load_dotenv
from src.ml.tr import ResumeNERPredictor
from src.db.pymongo import MongoConn
from src.helper.config.db_config import load_config

logging.basicConfig(level=logging.DEBUG)

load_dotenv()
app_port = os.getenv("APP_PORT")
minio_port = os.getenv("MINIO_API_PORT")
minio_host = os.getenv("MINIO_API_HOST")
minio_user = os.getenv("MINIO_ROOT_USER")
minio_password = os.getenv("MINIO_ROOT_PASSWORD")

model_path = "results/final_bert_ner_model.bin"

def serve():
    minio_client = MinioClient(
        endpoint=f"{minio_host}:{minio_port}",
        access_key=minio_user,
        secret_key=minio_password
    )

    ner_model = ResumeNERPredictor(model_path=model_path)

    db_config = load_config()

    mongo_conn = MongoConn(db_config)
    if not mongo_conn.connect_to_mongodb():
        raise Exception("failed to connect to MongoDB")
    
    service = SummarizerService(
        minio_client=minio_client,
        ner_model=ner_model,
        mongo_conn=mongo_conn
    )
    
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pb2_grpc.add_CVProcessorServiceServicer_to_server(
        service, server
    )

    server.add_insecure_port(f'0.0.0.0:{app_port}')
    server.start()
    print(f"ML service running on port {app_port}")
    server.wait_for_termination()

if __name__ == '__main__':
    serve()