FROM python:3.9-buster

WORKDIR /app

# Chỉ cài các gói hệ thống cần thiết
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Cài đặt các gói thiết yếu
COPY requirements.txt .
RUN pip install --no-cache-dir -U pip
RUN pip install --no-cache-dir spacy blis==0.7.9
RUN pip install --no-cache-dir -r requirements.txt

# Copy code và mô hình
COPY . .

# Chạy ứng dụng
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]