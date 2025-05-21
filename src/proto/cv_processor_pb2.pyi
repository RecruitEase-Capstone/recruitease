from google.protobuf import timestamp_pb2 as _timestamp_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class BatchPDFProcessRequest(_message.Message):
    __slots__ = ("bucket_name", "batch_id", "user_id", "pdf_files")
    BUCKET_NAME_FIELD_NUMBER: _ClassVar[int]
    BATCH_ID_FIELD_NUMBER: _ClassVar[int]
    USER_ID_FIELD_NUMBER: _ClassVar[int]
    PDF_FILES_FIELD_NUMBER: _ClassVar[int]
    bucket_name: str
    batch_id: str
    user_id: str
    pdf_files: _containers.RepeatedCompositeFieldContainer[PDFFileInfo]
    def __init__(self, bucket_name: _Optional[str] = ..., batch_id: _Optional[str] = ..., user_id: _Optional[str] = ..., pdf_files: _Optional[_Iterable[_Union[PDFFileInfo, _Mapping]]] = ...) -> None: ...

class PDFFileInfo(_message.Message):
    __slots__ = ("file_name", "size", "uploaded_at")
    FILE_NAME_FIELD_NUMBER: _ClassVar[int]
    SIZE_FIELD_NUMBER: _ClassVar[int]
    UPLOADED_AT_FIELD_NUMBER: _ClassVar[int]
    file_name: str
    size: int
    uploaded_at: _timestamp_pb2.Timestamp
    def __init__(self, file_name: _Optional[str] = ..., size: _Optional[int] = ..., uploaded_at: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...

class BatchPDFProcessResponse(_message.Message):
    __slots__ = ("batch_id", "total_files", "predictions")
    BATCH_ID_FIELD_NUMBER: _ClassVar[int]
    TOTAL_FILES_FIELD_NUMBER: _ClassVar[int]
    PREDICTIONS_FIELD_NUMBER: _ClassVar[int]
    batch_id: str
    total_files: int
    predictions: _containers.RepeatedCompositeFieldContainer[PredictionResult]
    def __init__(self, batch_id: _Optional[str] = ..., total_files: _Optional[int] = ..., predictions: _Optional[_Iterable[_Union[PredictionResult, _Mapping]]] = ...) -> None: ...

class PredictionResult(_message.Message):
    __slots__ = ("file_name", "prediction")
    FILE_NAME_FIELD_NUMBER: _ClassVar[int]
    PREDICTION_FIELD_NUMBER: _ClassVar[int]
    file_name: str
    prediction: CVPrediction
    def __init__(self, file_name: _Optional[str] = ..., prediction: _Optional[_Union[CVPrediction, _Mapping]] = ...) -> None: ...

class CVPrediction(_message.Message):
    __slots__ = ("name", "college_name", "degree", "graduation_year", "years_of_experience", "companies_worked_at", "designation", "skills", "location", "email_address")
    NAME_FIELD_NUMBER: _ClassVar[int]
    COLLEGE_NAME_FIELD_NUMBER: _ClassVar[int]
    DEGREE_FIELD_NUMBER: _ClassVar[int]
    GRADUATION_YEAR_FIELD_NUMBER: _ClassVar[int]
    YEARS_OF_EXPERIENCE_FIELD_NUMBER: _ClassVar[int]
    COMPANIES_WORKED_AT_FIELD_NUMBER: _ClassVar[int]
    DESIGNATION_FIELD_NUMBER: _ClassVar[int]
    SKILLS_FIELD_NUMBER: _ClassVar[int]
    LOCATION_FIELD_NUMBER: _ClassVar[int]
    EMAIL_ADDRESS_FIELD_NUMBER: _ClassVar[int]
    name: _containers.RepeatedScalarFieldContainer[str]
    college_name: _containers.RepeatedScalarFieldContainer[str]
    degree: _containers.RepeatedScalarFieldContainer[str]
    graduation_year: _containers.RepeatedScalarFieldContainer[str]
    years_of_experience: _containers.RepeatedScalarFieldContainer[str]
    companies_worked_at: _containers.RepeatedScalarFieldContainer[str]
    designation: _containers.RepeatedScalarFieldContainer[str]
    skills: _containers.RepeatedScalarFieldContainer[str]
    location: _containers.RepeatedScalarFieldContainer[str]
    email_address: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, name: _Optional[_Iterable[str]] = ..., college_name: _Optional[_Iterable[str]] = ..., degree: _Optional[_Iterable[str]] = ..., graduation_year: _Optional[_Iterable[str]] = ..., years_of_experience: _Optional[_Iterable[str]] = ..., companies_worked_at: _Optional[_Iterable[str]] = ..., designation: _Optional[_Iterable[str]] = ..., skills: _Optional[_Iterable[str]] = ..., location: _Optional[_Iterable[str]] = ..., email_address: _Optional[_Iterable[str]] = ...) -> None: ...

class FetchSummarizedPdfHistoryRequest(_message.Message):
    __slots__ = ("user_id",)
    USER_ID_FIELD_NUMBER: _ClassVar[int]
    user_id: str
    def __init__(self, user_id: _Optional[str] = ...) -> None: ...
