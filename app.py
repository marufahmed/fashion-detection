import os
import sys
import numpy as np
import skimage.io
import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
import matplotlib.pyplot as plt
from typing import Dict
import io
import shutil
from PIL import Image
import json

# Root directory of the project
ROOT_DIR = os.path.abspath(".")

# Import Mask RCNN
sys.path.append(ROOT_DIR)
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.config import Config

# Initialize FastAPI app
app = FastAPI(title="Fashion Detection API")

# Global variables
MODEL_DIR = os.path.join(ROOT_DIR, "models")
FASHION_MODEL_PATH = os.path.join(MODEL_DIR, "mask_rcnn_fashion_0008.h5") 
UPLOAD_DIR = os.path.join(ROOT_DIR, "uploads")
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")
IMAGE_SIZE = 512 

# Ensure directories exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Fashion class names
with open("label_descriptions.json") as f:
    label_descriptions = json.load(f)
class_names = ['BG'] + [x['name'] for x in label_descriptions['categories']]

class FashionConfig(Config):
    NAME = "fashion"
    NUM_CLASSES = len(class_names)  # Number of classes + background
    
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1  # Set to 1 for inference
    
    BACKBONE = 'resnet50'
    
    IMAGE_MIN_DIM = IMAGE_SIZE
    IMAGE_MAX_DIM = IMAGE_SIZE    
    IMAGE_RESIZE_MODE = 'none'
    
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)

class InferenceConfig(FashionConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

def resize_image(image_path):
    """
    Resize the image to match the model's input requirements while maintaining the aspect ratio.
    Adds padding to ensure the final size matches IMAGE_SIZE x IMAGE_SIZE.
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Get the original dimensions
    h, w, _ = image.shape

    # Calculate the scale to fit the IMAGE_SIZE while maintaining the aspect ratio
    scale = min(IMAGE_SIZE / h, IMAGE_SIZE / w)
    new_h = int(h * scale)
    new_w = int(w * scale)

    # Resize the image
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create a blank canvas of size IMAGE_SIZE x IMAGE_SIZE
    padded_image = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)

    # Center the resized image on the canvas
    y_offset = (IMAGE_SIZE - new_h) // 2
    x_offset = (IMAGE_SIZE - new_w) // 2
    padded_image[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_image

    return padded_image


def refine_masks(masks, rois):
    """Refine masks to prevent overlap"""
    areas = np.sum(masks.reshape(-1, masks.shape[-1]), axis=0)
    mask_index = np.argsort(areas)
    union_mask = np.zeros(masks.shape[:-1], dtype=bool)
    for m in mask_index:
        masks[:, :, m] = np.logical_and(masks[:, :, m], np.logical_not(union_mask))
        union_mask = np.logical_or(masks[:, :, m], union_mask)
    for m in range(masks.shape[-1]):
        mask_pos = np.where(masks[:, :, m]==True)
        if np.any(mask_pos):
            y1, x1 = np.min(mask_pos, axis=1)
            y2, x2 = np.max(mask_pos, axis=1)
            rois[m, :] = [y1, x1, y2, x2]
    return masks, rois

# Initialize the model at startup
config = InferenceConfig()
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
model.load_weights(FASHION_MODEL_PATH, by_name=True)

@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint"""
    return {"status": "healthy"}

@app.post("/detect")
async def detect_items(file: UploadFile = File(...)):
    """
    Endpoint to detect fashion objects in uploaded images.
    Returns both the visualization and detection results.
    """
    try:
        # Save uploaded file
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process image
        image = resize_image(file_path)
        
        # Run detection
        results = model.detect([image], verbose=1)
        r = results[0]
        
        # Refine masks if needed
        if r['masks'].size > 0:
            masks = np.zeros((image.shape[0], image.shape[1], r['masks'].shape[-1]), dtype=np.uint8)
            for m in range(r['masks'].shape[-1]):
                masks[:, :, m] = cv2.resize(r['masks'][:, :, m].astype('uint8'), 
                                            (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            masks, rois = refine_masks(masks, r['rois'])
        else:
            masks, rois = r['masks'], r['rois']
        
        # Create visualization
        output_path = os.path.join(OUTPUT_DIR, f"output_{file.filename}")
        plt.figure(figsize=(16, 16))
        visualize.display_instances(image, rois, masks, r['class_ids'], class_names, r['scores'])
        plt.savefig(output_path)
        plt.close()
        
        # Prepare detection results
        detections = []
        for i in range(len(r['class_ids'])):
            detections.append({
                'class': class_names[r['class_ids'][i]],
                'score': float(r['scores'][i]),
                'bbox': r['rois'][i].tolist()
            })
        
        # Clean up uploaded file
        os.remove(file_path)
        
        return {
            'detections': detections,
            'image_url': f"output_{file.filename}"
        }
    
    except Exception as e:
        # Clean up in case of error
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

# Constants for filtered classes
SLEEVE_ID = 31
NECKLINE_ID = 33
UPPERBODY_IDS = list(range(6))  # IDs 0-5 for upper body garments

def get_category_type(class_name: str, label_descriptions: dict) -> str:
    """
    Determine the category type (sleeve, neckline, or torso) based on the class name
    """
    # Direct matches for sleeve and neckline
    if class_name == "sleeve":
        return "sleeve"
    elif class_name == "neckline":
        return "neckline"
    
    # Check for torso items by finding the category in label descriptions
    for category in label_descriptions['categories']:
        if category['name'] == class_name and category['supercategory'] == "upperbody":
            return "torso"
    
    return "other"

@app.post("/detect-filtered")
async def detect_filtered_items(file: UploadFile = File(...)):
    """
    Endpoint to detect only sleeves, necklines, and upper body garments in uploaded images.
    Returns both the visualization and filtered detection results.
    """
    try:
        # Save uploaded file
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process image
        image = resize_image(file_path)
        
        # Run detection
        results = model.detect([image], verbose=1)
        r = results[0]
        
        # Get all detections first
        detections = []
        for i in range(len(r['class_ids'])):
            class_name = class_names[r['class_ids'][i]]
            category_type = get_category_type(class_name, label_descriptions)
            
            if category_type in ["sleeve", "neckline", "torso"]:
                detections.append({
                    'category_type': category_type,
                    'class_name': class_name,
                    'score': float(r['scores'][i]),
                    'bbox': r['rois'][i].tolist()
                })
        
        # Group detections by category type
        grouped_detections = {
            'sleeve': [],
            'neckline': [],
            'torso': []
        }
        
        for detection in detections:
            category = detection['category_type']
            if category in grouped_detections:
                grouped_detections[category].append({
                    'class': detection['class_name'],
                    'score': detection['score'],
                    'bbox': detection['bbox']
                })
        
        # Filter results for visualization
        filtered_indices = []
        for i, class_id in enumerate(r['class_ids']):
            class_name = class_names[class_id]
            category_type = get_category_type(class_name, label_descriptions)
            if category_type in ["sleeve", "neckline", "torso"]:
                filtered_indices.append(i)
        
        # Apply filters to all result arrays
        filtered_rois = r['rois'][filtered_indices]
        filtered_masks = r['masks'][..., filtered_indices] if r['masks'].size > 0 else r['masks']
        filtered_class_ids = r['class_ids'][filtered_indices]
        filtered_scores = r['scores'][filtered_indices]
        
        # Create visualization
        output_path = os.path.join(OUTPUT_DIR, f"filtered_output_{file.filename}")
        plt.figure(figsize=(16, 16))
        if len(filtered_indices) > 0:
            visualize.display_instances(image, filtered_rois, filtered_masks, 
                                    filtered_class_ids, class_names, 
                                    filtered_scores)
        else:
            plt.imshow(image)
        plt.savefig(output_path)
        plt.close()
        
        # Clean up uploaded file
        os.remove(file_path)
        
        return {
            'grouped_detections': grouped_detections,
            'image_url': f"filtered_output_{file.filename}"
        }
    
    except Exception as e:
        # Clean up in case of error
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")
    
    
@app.get("/outputs/{filename}")
async def get_output(filename: str):
    """Endpoint to retrieve output images"""
    file_path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(file_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)