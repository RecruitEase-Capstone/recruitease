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

logging.basicConfig(level=logging.INFO)

load_dotenv()
app_port = os.getenv("APP_PORT")
minio_port = os.getenv("MINIO_API_PORT")
minio_host = os.getenv("MINIO_API_HOST")
minio_user = os.getenv("MINIO_ROOT_USER")
minio_password = os.getenv("MINIO_ROOT_PASSWORD")

model_path = "results/model-state.bin"

def serve():
    try: 
        minio_client = MinioClient(
            endpoint=f"{minio_host}:{minio_port}",
            access_key=minio_user,
            secret_key=minio_password
        )
    except Exception as e:
        raise Exception(f"failed to initiate minio: {e}")

    ner_model = ResumeNERPredictor(model_path=model_path)
    if ner_model is None:
        logging.fatal("NER model is None!")
        raise Exception("NER model is None!")
    else:
        logging.info("NER model instance created successfully.")

    db_config = load_config()

    logging.info("Connecting to MongoDB...")
    mongo_conn = MongoConn(db_config)
    if not mongo_conn.connect_to_mongodb():
        logging.fatal("Failed to connect to MongoDB!")
        raise Exception("failed to connect to MongoDB")
    logging.info("Successfully connected to MongoDB.")

    logging.info("initiating summarize service...")
    try:
        service = SummarizerService(
            minio_client=minio_client,
            ner_model=ner_model,
            mongo_conn=mongo_conn
        )
    except Exception as e:
        logging.fatal(f"error while initiate summarize service: {e}")
        raise e 
    
    logging.info("success initiate summarize service")

    logging.info("start initiaate grpc server")
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pb2_grpc.add_CVProcessorServiceServicer_to_server(
        service, server
    )

    server.add_insecure_port(f'0.0.0.0:{app_port}')
    server.start()
    logging.info(f"ML service running on port {app_port}")
    server.wait_for_termination()

if __name__ == '__main__':
    serve()