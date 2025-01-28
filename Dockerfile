FROM python:3.7-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system dependencies all in one layer to reduce image size
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    curl \
    wget \
    git \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    libglib2.0-0 \
    zlib1g-dev \
    libjpeg-dev \
    libpng-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create working directory and copy requirements
WORKDIR /app
COPY requirements.txt .

# Add FastAPI and uvicorn to requirements
RUN echo "fastapi==0.68.1\nuvicorn==0.15.0\npython-multipart==0.0.5" >> requirements.txt

# Use BuildKit cache mount to speed up pip
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install opencv-python-headless && \
    pip install -r requirements.txt

# Clone repositories and install packages all in one layer
RUN git clone https://github.com/akTwelve/Mask_RCNN.git && \
    cd Mask_RCNN && \
    python setup.py install && \
    git clone https://github.com/waleedka/coco.git && \
    cd coco/PythonAPI && \
    python setup.py build_ext --inplace && \
    python setup.py install

# Download pre-trained weights
RUN mkdir -p /app/Mask_RCNN/logs && \
    wget -O /app/Mask_RCNN/mask_rcnn_coco.h5 https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5

# Create required directories
RUN mkdir -p /app/Mask_RCNN/uploads /app/Mask_RCNN/outputs

# Copy application files
COPY app.py /app/Mask_RCNN/
COPY label_descriptions.json /app/Mask_RCNN
COPY models/* /app/Mask_RCNN/models/

# Set working directory and command
WORKDIR /app/Mask_RCNN
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]