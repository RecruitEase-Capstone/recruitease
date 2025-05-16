import os
import tempfile
import grpc
import traceback
import proto.cv_processor_pb2 as pb2
import proto.cv_processor_pb2_grpc as pb2_grpc

from typing import Optional
from minio import Minio
from ml.tr import ResumeNERPredictor

class SummarizerService(pb2_grpc.CVProcessorServiceServicer):
    def __init__(self, minio_client, ner_model: ResumeNERPredictor):
        self.minio_client = minio_client
        self.ner_model = ner_model

    def ProcessBatchPDF(self, request, context):
        try:
            print(">> start processing the batch pdf")
            batch_id = request.batch_id
            bucket = request.bucket_name
            
            # Validate request parameters
            if not batch_id:
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Batch ID is required")
            if not bucket:
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Bucket name is required")
            if not request.pdf_files:
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, "No PDF files provided")
                
            predictions = []

            for pdf_info in request.pdf_files:
                file_name = pdf_info.file_name
                temp_path = None

                try:
                    print(f"Processing file: {file_name}")
                    
                    try:
                        temp_path = self.minio_client.download_file(bucket, file_name)
                        print(f"File downloaded to: {temp_path}")
                    except Exception as e:
                        print(f"MinIO error for {file_name}: {str(e)}")
                        context.abort(
                            grpc.StatusCode.NOT_FOUND if "not found" in str(e).lower() 
                            else grpc.StatusCode.UNAVAILABLE,
                            f"Failed to download file {file_name}: {str(e)}"
                        )
                    
                    try:
                        extracted = self.ner_model.predict_from_pdf(temp_path)
                        print(f"Prediction complete for: {file_name}")
                    except Exception as e:
                        print(f"NER model error for {file_name}: {str(e)}")
                        traceback.print_exc()
                        context.abort(
                            grpc.StatusCode.INTERNAL,
                            f"Failed to process file {file_name}: {str(e)}"
                        )
                    
                    pred = pb2.CVPrediction(
                        name=extracted.get("Name", ""),
                        college_name=extracted.get("College Name", ""),
                        degree=extracted.get("Degree", ""),
                        graduation_year=extracted.get("Graduation Year", ""),
                        years_of_experience=extracted.get("Years of Experience", ""),
                        companies_worked_at=extracted.get("Companies worked at", ""),
                        designation=extracted.get("Designation", ""),
                        skills=extracted.get("Skills", ""),
                        location=extracted.get("Location", ""),
                        email_address=extracted.get("Email Address", "")
                    )
                    
                    predictions.append(pb2.PredictionResult(
                        file_name=file_name,
                        prediction=pred
                    ))
                    print(f"Successfully processed: {file_name}")
                    
                except Exception as e:
                    err_msg = f"Failed processing {file_name}: {str(e)}"
                    print(err_msg)
                    traceback.print_exc()
                    
                finally:
                    if temp_path and os.path.exists(temp_path):
                        try:
                            os.remove(temp_path)
                            print(f"Removed temporary file: {temp_path}")
                        except Exception as e:
                            print(f"Error removing temp file: {str(e)}")

            print(f"Returning {len(predictions)} predictions")
            return pb2.BatchPDFProcessResponse(
                batch_id=batch_id,
                total_files=len(predictions),
                predictions=predictions
            )
            
        except Exception as e:
            print(f"Critical error in ProcessBatchPDF: {str(e)}")
            traceback.print_exc()
            context.abort(self._map_exception_to_grpc_code(e), f"Service error: {str(e)}")
            
    def _map_exception_to_grpc_code(self, exception: Exception) -> grpc.StatusCode:
        exception_str = str(exception).lower()
        exception_type = type(exception).__name__
        
        if any(term in exception_str for term in ["timeout", "deadline", "timed out"]):
            return grpc.StatusCode.DEADLINE_EXCEEDED
        
        if any(term in exception_str for term in ["connection", "network", "unreachable"]):
            return grpc.StatusCode.UNAVAILABLE
            
        if any(term in exception_str for term in ["not found", "no such", "404"]):
            return grpc.StatusCode.NOT_FOUND
            
        if any(term in exception_str for term in ["permission", "unauthorized", "forbidden", "access denied"]):
            return grpc.StatusCode.PERMISSION_DENIED
            
        if any(term in exception_str for term in ["invalid", "argument", "parameter"]):
            return grpc.StatusCode.INVALID_ARGUMENT
            
        if "resource exhausted" in exception_str or "quota" in exception_str:
            return grpc.StatusCode.RESOURCE_EXHAUSTED
            
        if exception_type in ["FileNotFoundError", "NotFoundError"]:
            return grpc.StatusCode.NOT_FOUND
            
        if exception_type in ["ValueError", "TypeError", "AttributeError"]:
            return grpc.StatusCode.INVALID_ARGUMENT
            
        if exception_type in ["PermissionError"]:
            return grpc.StatusCode.PERMISSION_DENIED
            
        return grpc.StatusCode.INTERNAL
