import os
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from flask import Flask, request, jsonify, render_template, url_for, send_from_directory
from pathlib import Path

# --- Configuration ---
# Point to your EfficientNet models folder
MODELS_DIR = Path("trained_models") 
CLASSES_DIR = MODELS_DIR / "class_indices"
SOLUTIONS_PATH = MODELS_DIR / "solutions_db.json"
UPLOAD_FOLDER = "uploads"
MODEL_ARCH = "efficientnet_b0" # We will use this model set

# --- 1. Initialize Flask App ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- 2. Load Models, Classes, and Solutions (Done Once) ---
print("Loading all models and databases... This may take a moment.")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define image size
if MODEL_ARCH == "resnet50":
    input_size = 224
else:
    input_size = 256 # EfficientNet default

# Define the image transforms (must match validation/test)
preprocess_transform = transforms.Compose([
    transforms.Resize(input_size + 32),
    transforms.CenterCrop(input_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load solutions database
with open(SOLUTIONS_PATH, 'r', encoding='utf-8') as f:
    solutions_db = json.load(f)
print(f"Loaded solutions from {SOLUTIONS_PATH}")

# Dictionary to hold all our loaded models and class lists
models_dict = {}
crop_list = []

# Loop through model files in the directory
for model_path in MODELS_DIR.glob('model_*.pth'):
    try:
        # e.g., "model_banana.pth" -> "banana"
        # This part assumes your filenames are like 'model_banana.pth'
        # If they are 'model_resnet_banana.pth', this logic will fail.
        # We are assuming the EfficientNet models.
        
        # Get the crop key from the filename
        filename_parts = model_path.stem.split('_')
        if filename_parts[0] != 'model' or len(filename_parts) < 2:
            print(f"Skipping unknown model file: {model_path.name}")
            continue
        
        # Assumes format 'model_banana.pth' or 'model_resnet_banana.pth'
        crop_key = filename_parts[-1] # Gets "banana"
        
        # e.g., "classes_banana.json"
        # We need to find the matching classes file
        classes_path = CLASSES_DIR / f"classes_{crop_key}.json"
        
        # Check if the ResNet classes file exists if this is a ResNet model
        if "resnet" in model_path.stem:
             classes_path = MODELS_DIR / "class_indices_resnet" / f"classes_resnet_{crop_key}.json"

        if not classes_path.exists():
            print(f"Warning: No class file found at {classes_path} for model {model_path.name}. Skipping.")
            continue
            
        with open(classes_path, 'r') as f:
            class_names = json.load(f)
        num_classes = len(class_names)
        
        # Load the model architecture
        if MODEL_ARCH == "resnet50":
            model = models.resnet50()
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, num_classes)
        else:
            model = models.efficientnet_b0()
            num_ftrs = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_ftrs, num_classes)
            
        # Load the trained weights
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval() # Set to evaluation mode
        
        # Store the model and its class names
        models_dict[crop_key] = {
            "model": model,
            "classes": class_names,
            "full_folder_name": f"Balanced {crop_key.capitalize()} Dataset"
        }
        
        # Add to dropdown list
        crop_list.append({
            "key": crop_key, # "banana"
            "name": crop_key.capitalize() # "Banana"
        })
        
        print(f"Successfully loaded model and classes for: {crop_key}")

    except Exception as e:
        print(f"Error loading model for {model_path.name}: {e}")

print("--- All models loaded successfully! ---")


# --- 3. Prediction Function ---
def predict_image(image_path, crop_key):
    """
    Takes an image path and a CROP KEY, runs it through the
    correct model, and returns Top 3 predictions.
    """
    if crop_key not in models_dict:
        print(f"Error: No model found for crop_key '{crop_key}'")
        return None
        
    model_info = models_dict[crop_key]
    model = model_info["model"]
    class_names = model_info["classes"]
    base_class_path = model_info["full_folder_name"]

    try:
        image = Image.open(image_path).convert("RGB")
        input_tensor = preprocess_transform(image)
        input_batch = input_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_batch)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
        
        top_probs, top_indices = torch.topk(probabilities, 3)
        
        predictions = []
        for i in range(top_probs.size(0)):
            confidence = top_probs[i].item() * 100
            class_index = top_indices[i].item()
            
            disease_class_name = class_names[class_index]
            full_class_name = f"{base_class_path}/{disease_class_name}"
            
            # Get just the friendly name from the solution DB
            try:
                disease_name = solutions_db[full_class_name]["disease_name"]
            except KeyError:
                # This fallback is for keys that might have case-mismatch, like "healthy"
                disease_name = disease_class_name.replace('_', ' ') 
            
            predictions.append({
                "full_class_name": full_class_name,
                "disease_name": disease_name,
                "confidence": f"{confidence:.2f}"
            })
            
        return predictions

    except Exception as e:
        print(f"Error during prediction: {e}")
        return None

# --- 4. Flask Routes ---

@app.route('/', methods=['GET'])
def home():
    """Renders the main landing page."""
    return render_template('index.html')

@app.route('/app', methods=['GET'])
def launch_app():
    """Renders the main diagnosis app page."""
    sorted_crops = sorted(crop_list, key=lambda x: x['name'])
    return render_template('app.html', crops=sorted_crops)

@app.route('/predict', methods=['POST'])
def handle_prediction():
    """Handles the image upload, predicts, and returns results as JSON."""
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    crop_key = request.form.get('crop_key') # Get the crop from the dropdown
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if not crop_key:
        return jsonify({"error": "No crop selected"}), 400

    if file:
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        predictions = predict_image(filepath, crop_key)
        
        if predictions:
            return jsonify({
                "predictions": predictions,
                "image_url": url_for('send_uploaded_file', filename=filename)
            })
        else:
            return jsonify({"error": "Could not process image"}), 500

@app.route('/solution', methods=['POST'])
def get_solution():
    """Fetches the solution from the JSON database based on the class name."""
    class_name = request.json.get('class_name')
    if not class_name:
        return jsonify({"error": "Missing class name"}), 400
    
    print(f"\n[DEBUG] App received request for class: '{class_name}'")
    
    solution = solutions_db.get(class_name) 
    
    if solution:
        print("[DEBUG] Solution found.")
        return jsonify(solution)
    else:
        # Check for case mismatches, e.g. "healthy" vs "Healthy"
        for key in solutions_db.keys():
            if key.lower() == class_name.lower():
                print(f"[DEBUG] Found case-mismatch. Returning solution for: {key}")
                return jsonify(solutions_db[key])
        
        print(f"[DEBUG] *** SOLUTION NOT FOUND in solutions_db.json for key: '{class_name}' ***")
        return jsonify({"error": f"Solution not found for class: {class_name}"}), 404

@app.route('/uploads/<filename>')
def send_uploaded_file(filename):
    """Serves the uploaded image file."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# --- 5. Run the App ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)