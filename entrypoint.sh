#!/bin/bash
echo "Khởi động ứng dụng, đang tải mô hình..."
python -c "import app; print('Mô hình đã được tải')"
exec gunicorn --bind 0.0.0.0:5000 --timeout 300 --workers 1 --threads 2 app:app