FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Set environment variable to allow execstack
ENV DOCKER_BUILD=1

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

# Disable SELinux execstack requirement
CMD ["sh", "-c", "execstack -c /usr/local/lib/python3.11/site-packages/onnxruntime/capi/*.so 2>/dev/null || true && gunicorn --bind 0.0.0.0:5000 --workers 1 --timeout 120 app:app"]