# Backend Server - Diagnodash AI

import os
import torch
import cv2
import numpy as np
import base64
import torchvision.transforms as transforms
from flask import Flask, request, jsonify
from PIL import Image
import torchvision.ops as ops

# Import needed modules
from model import MobileNetRefineDetLiteCBAM
from utils import AnchorGenerator, decode

app = Flask(__name__)

# 
MODEL_PATH = 'model.pth'
IMG_SIZE = 512
CONF_THRESHOLD = 0.40
IOU_THRESHOLD = 0.30

# Classes
CLASSES = ['__background__', 'battery_icon', 'engine_icon', 'oil_pressure_icon', 'parking_brake_icon', 'power_steering_icon']

# Colors for Bounding Boxes
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# We use CPU for the demo
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"⚙️ Running on Device: {DEVICE}")

# Loading the Model weight file
print(f"🚀 Loading Model...")
model = MobileNetRefineDetLiteCBAM(num_classes=6).to(DEVICE)

# Check if the model is available 
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE)) # Load weights and set to Evaluation 
    model.eval()
    print("✅ Model Loaded Successfully!")
else:
    print(f"❌ Error: {MODEL_PATH} not found.")

# Generate Anchors
# RefineDet requires pre-generated anchor boxes for object localization
anchor_gen = AnchorGenerator(IMG_SIZE)
anchors = anchor_gen.forward(DEVICE)

# Transform (Same as training)
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)), # Standardizes input images to match the format used during training
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Normalization using ImageNet mean/std values
])

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    
    try:
        # Reading an image using OpenCV2
        file_bytes = np.frombuffer(file.read(), np.uint8)
        orig_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        h_orig, w_orig, _ = orig_image.shape

        # Convert to PIL for Transform for PyTorch compatibility
        img_pil = Image.fromarray(cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB))
        # Apply transforms
        input_tensor = transform(img_pil).unsqueeze(0).to(DEVICE)

        # Inference
        with torch.no_grad():
            # RefineDet returns outputs for ARM and ODM (Object Detection Module)
            arm_loc, arm_conf, odm_loc, odm_conf = model(input_tensor)
            
            odm_conf = torch.nn.functional.softmax(odm_conf, dim=2) # Apply Softmax to get probability scores
            # Decode box offsets into real coordinates
            boxes = decode(odm_loc[0], anchors)
            
            # Scale boxes back to original image size
            boxes[:, 0::2] *= w_orig
            boxes[:, 1::2] *= h_orig

        scores, labels = torch.max(odm_conf[0], dim=1) # Get highest confidence score for each anchor

        # Filter out low confidence detections
        mask = scores > CONF_THRESHOLD
        boxes = boxes[mask]
        scores = scores[mask]
        labels = labels[mask]

        # Remove 'Background' class detections
        mask_bg = labels > 0
        boxes = boxes[mask_bg]
        scores = scores[mask_bg]
        labels = labels[mask_bg]

        detected_class_name = "default"
        best_confidence = 0.0

        if boxes.size(0) > 0:
            # Removes overlapping boxes referring to the same object using NMS
            keep_idx = ops.nms(boxes, scores, IOU_THRESHOLD)
            
            # Keep only the best boxes
            final_boxes = boxes[keep_idx]
            final_scores = scores[keep_idx]
            final_labels = labels[keep_idx]

            # Get the best detection for text response
            best_idx = 0 
            detected_class_name = CLASSES[final_labels[best_idx].item()]
            best_confidence = float(final_scores[best_idx].item())

            # Bounding Box Drawing for every box
            for i in range(final_boxes.size(0)):
                box = final_boxes[i].cpu().numpy().astype(int)
                score = final_scores[i].item()
                label_idx = final_labels[i].item()
                
                label_name = CLASSES[label_idx]
                color = COLORS[label_idx]
                
                # Draw Box
                cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), color, 3)
                
                # Draw Label Text with Score
                text = f"{label_name} {score:.2f}"
                cv2.putText(orig_image, text, (box[0], box[1]-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # Convert the annotated OpenCV image to Base64 string
            _, buffer = cv2.imencode('.jpg', orig_image)
            img_str = base64.b64encode(buffer).decode('utf-8')
            result_image_base64 = f"data:image/jpeg;base64,{img_str}"

            return jsonify({ # Sending the detected object
                "success": True,
                "detected_class": detected_class_name,
                "confidence": best_confidence,
                "result_image": result_image_base64
            })
        else: ## Handle case where no object is found
            return jsonify({
                "success": False,
                "message": "No specific warning light detected."
            })

    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({"error": str(e)}), 500

# Server Start
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
