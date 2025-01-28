# Clone this repository
```
git clone https://github.com/marufahmed/fashion-detection.git
```
# Install git Large File System
```
git lfs install
```
# Pull models with lfs
```
git lfs pull
```
# Build docker container
```
docker build -t fashion_detection_api .
```
# Run containerized application
```
docker run -p 8000:8000 fashion_detection_api
```