# Clone this repository
```
git clone https://github.com/marufahmed/fashion-detection.git
```
# Build docker container
```
docker build -t fashion_detection_api .
```
# Run containerized application
```
docker run -p 8000:8000 fashion_detection_api
```

# Fashion Detection API Documentation

## Base URL
```
http://localhost:8000
```

## Endpoints

### 1. Health Check
Check if the API is running.

```
GET /health
```

#### Response
```json
{
    "status": "healthy"
}
```

### 2. General Fashion Detection
Detect all fashion items in an image.

```
POST /detect
```

#### Request
- Content-Type: `multipart/form-data`
- Body: 
  - `file`: Image file (supported formats: JPG, PNG)

#### Response
```json
{
    "detections": [
        {
            "class": string,        // Class name of detected item
            "score": float,         // Confidence score (0-1)
            "bbox": [int, int, int, int]  // Bounding box coordinates [y1, x1, y2, x2]
        }
    ],
    "image_url": string  // URL to access the annotated image
}
```

#### Example Response
```json
{
    "detections": [
        {
            "class": "sleeve",
            "score": 0.927,
            "bbox": [178, 319, 386, 384]
        },
        {
            "class": "neckline",
            "score": 0.737,
            "bbox": [152, 220, 184, 292]
        }
    ],
    "image_url": "output_image.jpg"
}
```

### 3. Filtered Fashion Detection
Detect only sleeves, necklines, and torso garments.

```
POST /detect-filtered
```

#### Request
- Content-Type: `multipart/form-data`
- Body: 
  - `file`: Image file (supported formats: JPG, PNG)

#### Response
```json
{
    "grouped_detections": {
        "sleeve": [
            {
                "class": string,    // Class name of detected item
                "score": float,     // Confidence score (0-1)
                "bbox": [int, int, int, int]  // Bounding box coordinates [y1, x1, y2, x2]
            }
        ],
        "neckline": [...],         // Same structure as sleeve
        "torso": [...]             // Same structure as sleeve
    },
    "image_url": string  // URL to access the annotated image
}
```

#### Example Response
```json
{
    "grouped_detections": {
        "sleeve": [
            {
                "class": "sleeve",
                "score": 0.927,
                "bbox": [178, 319, 386, 384]
            }
        ],
        "neckline": [
            {
                "class": "neckline",
                "score": 0.737,
                "bbox": [152, 220, 184, 292]
            }
        ],
        "torso": []
    },
    "image_url": "filtered_output_image.jpg"
}
```

### 4. Get Output Image
Retrieve the processed image with annotations.

```
GET /outputs/{filename}
```

#### Parameters
- `filename`: Name of the output image file (returned in image_url from detection endpoints)

#### Response
- Content-Type: `image/jpeg` or `image/png`
- Body: Binary image data

## Notes
- Detection confidence scores range from 0 to 1, where 1 indicates highest confidence
- Bounding box coordinates are in format [y1, x1, y2, x2] where (x1,y1) is the top-left corner and (x2,y2) is the bottom-right corner
- Filtered detection only includes:
  - Sleeves (garment parts)
  - Necklines (garment parts)
  - Torso items (upperbody supercategory)
- Input images are automatically resized to maintain aspect ratio while fitting within the model's required dimensions
