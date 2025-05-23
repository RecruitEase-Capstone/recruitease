FROM pytorch/pytorch:2.7.0-cuda11.8-cudnn9-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 50052

CMD [ "python", "-m", "src.main" ]