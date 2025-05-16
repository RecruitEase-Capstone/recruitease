generate-proto:
	python -m grpc_tools.protoc -Isrc/proto --python_out=./src/proto/ --pyi_out=./src/proto/ --grpc_python_out=src/proto/ src/proto/cv_processor.proto

main-run:
	python -m src.main

.PHONY:
	generate-proto