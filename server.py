import os
import io
import torch
import torchvision.transforms as transforms
from PIL import Image
import torchvision.ops as ops
from flask import Flask, request, jsonify

# --- IMPORT YOUR CUSTOM MODULES ---
# These imports work because model.py and utils.py are in the same folder
from model import MobileNetRefineDetLiteCBAM
from utils import AnchorGenerator, decode

app = Flask(__name__)

# --- CONFIGURATION ---
MODEL_PATH = 'model.pth'  # You renamed recall_repair_epoch_10.pth to this
IMG_SIZE = 512
CONF_THRESHOLD = 0.40     # Only show strong results
IOU_THRESHOLD = 0.30      # Overlap threshold for filtering

# Class mapping (Must match your data.py training order exactly)
CLASSES = [
    '__background__', 
    'battery_icon', 
    'engine_icon', 
    'oil_pressure_icon', 
    'parking_brake_icon', 
    'power_steering_icon'
]

# --- 1. SETUP DEVICE ---
# This ensures it works on Cloud Servers (CPU) and your Laptop (GPU)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"⚙️  Server starting on device: {DEVICE}")

# --- 2. LOAD MODEL ---
print(f"🚀 Loading Model from {MODEL_PATH}...")
model = MobileNetRefineDetLiteCBAM(num_classes=6).to(DEVICE)

if os.path.exists(MODEL_PATH):
    # map_location ensures the model loads onto CPU if GPU is missing
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("✅ Model weights loaded successfully!")
else:
    print(f"❌ CRITICAL ERROR: {MODEL_PATH} not found in current directory.")

# --- 3. GENERATE ANCHORS ---
# We generate these once at startup to save time
anchor_gen = AnchorGenerator(IMG_SIZE)
anchors = anchor_gen.forward(DEVICE)

# --- 4. PREPROCESSING ---
# Must match the transforms used in your training
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.route('/', methods=['GET'])
def home():
    return "Diagnodash AI Server is Awake and Ready!", 200

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    
    try:
        # A. Read Image
        img_bytes = file.read()
        img_pil = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        w_orig, h_orig = img_pil.size

        # B. Preprocess
        input_tensor = transform(img_pil).unsqueeze(0).to(DEVICE)

        # C. Inference
        with torch.no_grad():
            arm_loc, arm_conf, odm_loc, odm_conf = model(input_tensor)
            
            # Convert confidence to probabilities
            odm_conf = torch.nn.functional.softmax(odm_conf, dim=2)
            
            # Decode boxes
            boxes = decode(odm_loc[0], anchors)
            
            # Scale boxes back to original image size
            boxes[:, 0::2] *= w_orig
            boxes[:, 1::2] *= h_orig

        # D. Process Results
        scores, labels = torch.max(odm_conf[0], dim=1)

        # Filter: Confidence Threshold
        mask = scores > CONF_THRESHOLD
        boxes = boxes[mask]
        scores = scores[mask]
        labels = labels[mask]

        # Filter: Remove Background Class (0)
        mask_bg = labels > 0
        boxes = boxes[mask_bg]
        scores = scores[mask_bg]
        labels = labels[mask_bg]

        if boxes.size(0) > 0:
            # E. Non-Maximum Suppression (NMS)
            # Removes duplicate boxes for the same icon
            keep_idx = ops.nms(boxes, scores, IOU_THRESHOLD)
            
            # We take the single best detection
            best_idx = keep_idx[0]
            
            detected_label = CLASSES[labels[best_idx].item()]
            confidence = float(scores[best_idx].item())
            
            return jsonify({
                "success": True,
                "detected_class": detected_label,
                "confidence": confidence
            })
        else:
            return jsonify({
                "success": False,
                "message": "No warning light detected."
            })

    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Render and other clouds usually provide a PORT env variable
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)