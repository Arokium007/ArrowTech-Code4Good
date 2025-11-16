import os
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from flask import Flask, request, jsonify, render_template, url_for, send_from_directory, abort
from pathlib import Path
import requests
import io
import uuid 
import csv 

# --- Configuration ---
# Paths to your two model sets
MODELS_DIR_EFFICIENTNET = Path("trained_models")
CLASSES_DIR_EFFICIENTNET = MODELS_DIR_EFFICIENTNET / "class_indices"

MODELS_DIR_RESNET = Path("trained_models_resnet")
CLASSES_DIR_RESNET = MODELS_DIR_RESNET / "class_indices_resnet"

SOLUTIONS_PATH = MODELS_DIR_EFFICIENTNET / "solutions_db.json" # Assuming DB is in the first folder
UPLOAD_FOLDER = "uploads"

# --- NEW: Admin Database Files ---
ADMIN_DB_PATH = Path("admin_review_db.json")
CORRECTED_CSV_PATH = Path("corrected_labels.csv")

# --- 1. Initialize Flask App ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# --- Initialize Admin DB files if they don't exist ---
if not ADMIN_DB_PATH.exists():
    with open(ADMIN_DB_PATH, 'w') as f:
        json.dump([], f) # Create an empty list
if not CORRECTED_CSV_PATH.exists():
    with open(CORRECTED_CSV_PATH, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["image_url", "corrected_crop", "corrected_disease"])





# --- WhatsApp Configuration (Hardcoded) ---
WHATSAPP_TOKEN = "EAAMNywOQTwkBP8maZAdR8nCqHQEENsgD2APjKOoxcx9ef2f1W2fMFEDbJ8GKCU6bAwMBZAkiBi3iFFso6YrQrQ0e6U8YtUte6oLZBl0ZBCoK6sWZCMgYdlWVZBCkJpFZCYEd3Be0UEUBvGXOroHnZCLOZBIFsHAUJfq2HNLfUBFkRAZBLqQaoVykZAfuqOMnlDO5HIDR2PrKQZAjqUyZABZAb47zUOzInoxdhkBA1hnSZCYxAzlceAZCOGZBS2gXme9EtYjLfNQZDZD"
VERIFY_TOKEN = "ydukfTFYYGIG8562rdtv525hutvUGY" # Your secret password
PHONE_NUMBER_ID = "424729797382512"
# -------------------------------------------
WHATSAPP_API_URL = f"https://graph.facebook.com/v22.0/{PHONE_NUMBER_ID}/messages"
# -------------------------------------------


# --- 2. Load Models, Classes, and Solutions (Done Once) ---
print("Loading all models and databases... This may take a moment.")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define image transforms for BOTH models
preprocess_transform_effnet = transforms.Compose([
    transforms.Resize(256 + 32),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
preprocess_transform_resnet = transforms.Compose([
    transforms.Resize(224 + 32),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

with open(SOLUTIONS_PATH, 'r', encoding='utf-8') as f:
    solutions_db = json.load(f)
print(f"Loaded solutions from {SOLUTIONS_PATH}")

models_dict = {}
crop_list = []
ALL_CROP_KEYS = ["banana", "cucumber", "eggplant", "litchi", "mangoo", "onion", "papaya", "potato", "pumpkin", "tomato"]

for crop_key in ALL_CROP_KEYS:
    # ... (Model loading logic remains identical to your file) ...
    print(f"--- Loading models for: {crop_key} ---")
    model_effnet = None
    model_resnet = None
    class_names_effnet = None
    class_names_resnet = None

    # --- 1. Load EfficientNet Model ---
    model_path_effnet = MODELS_DIR_EFFICIENTNET / f"model_{crop_key}.pth"
    classes_path_effnet = CLASSES_DIR_EFFICIENTNET / f"classes_{crop_key}.json"

    if model_path_effnet.exists() and classes_path_effnet.exists():
        try:
            with open(classes_path_effnet, 'r') as f:
                class_names_effnet = json.load(f)
            num_classes = len(class_names_effnet)

            model_effnet = models.efficientnet_b0()
            num_ftrs = model_effnet.classifier[1].in_features
            model_effnet.classifier[1] = nn.Linear(num_ftrs, num_classes)
            
            model_effnet.load_state_dict(torch.load(model_path_effnet, map_location=device))
            model_effnet = model_effnet.to(device)
            model_effnet.eval()
            print(f"Loaded EfficientNet model for {crop_key}")
        except Exception as e:
            print(f"Error loading EfficientNet for {crop_key}: {e}")
            model_effnet = None
    else:
        print(f"EfficientNet model or class file for {crop_key} not found.")

    # --- 2. Load ResNet50 Model ---
    model_path_resnet = MODELS_DIR_RESNET / f"model_resnet_{crop_key}.pth"
    classes_path_resnet = CLASSES_DIR_RESNET / f"classes_resnet_{crop_key}.json"

    if model_path_resnet.exists() and classes_path_resnet.exists():
        try:
            with open(classes_path_resnet, 'r') as f:
                class_names_resnet = json.load(f)
            num_classes = len(class_names_resnet)

            model_resnet = models.resnet50()
            num_ftrs = model_resnet.fc.in_features
            model_resnet.fc = nn.Linear(num_ftrs, num_classes)
            
            model_resnet.load_state_dict(torch.load(model_path_resnet, map_location=device))
            model_resnet = model_resnet.to(device)
            model_resnet.eval()
            print(f"Loaded ResNet50 model for {crop_key}")
        except Exception as e:
            print(f"Error loading ResNet50 for {crop_key}: {e}")
            model_resnet = None
    else:
        print(f"ResNet50 model or class file for {crop_key} not found.")

    # --- 3. Validate and Store ---
    if model_effnet and model_resnet:
        if class_names_effnet != class_names_resnet:
            print(f"CRITICAL ERROR: Class mismatch for {crop_key}. ResNet will be disabled for this crop.")
            model_resnet = None
        else:
            print(f"Ensemble enabled for {crop_key}.")
    
    if model_effnet or model_resnet:
        models_dict[crop_key] = {
            "model_effnet": model_effnet,
            "model_resnet": model_resnet,
            "classes": class_names_effnet or class_names_resnet, 
            "full_folder_name": f"Balanced {crop_key.capitalize()} Dataset"
        }
        crop_list.append({
            "key": crop_key, 
            "name": crop_key.capitalize()
        })
    else:
        print(f"Warning: No models could be loaded for {crop_key}.")

print("--- All models loaded successfully! ---")


# --- 3. AI Prediction Function (MODIFIED FOR ENSEMBLE) ---
def predict_image(image_bytes, crop_key):
    if crop_key not in models_dict:
        print(f"Error: No model found for crop_key '{crop_key}'")
        return None
        
    model_info = models_dict[crop_key]
    model_effnet = model_info.get("model_effnet")
    model_resnet = model_info.get("model_resnet")
    class_names = model_info["classes"]
    base_class_path = model_info["full_folder_name"]
    model_count = 0
    
    probs_effnet = torch.zeros(len(class_names), device=device)
    probs_resnet = torch.zeros(len(class_names), device=device)

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        if model_effnet:
            input_tensor_effnet = preprocess_transform_effnet(image).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model_effnet(input_tensor_effnet)
                probs_effnet = torch.nn.functional.softmax(output[0], dim=0)
            model_count += 1

        if model_resnet:
            input_tensor_resnet = preprocess_transform_resnet(image).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model_resnet(input_tensor_resnet)
                probs_resnet = torch.nn.functional.softmax(output[0], dim=0)
            model_count += 1

        if model_count == 0:
            print(f"No models were available to predict for {crop_key}")
            return None

        final_probabilities = (probs_effnet + probs_resnet) / model_count
        top_probs, top_indices = torch.topk(final_probabilities, 3)
        
        predictions = []
        for i in range(top_probs.size(0)):
            confidence = top_probs[i].item() * 100
            class_index = top_indices[i].item()
            disease_class_name = class_names[class_index]
            full_class_name = f"{base_class_path}/{disease_class_name}"
            
            try:
                disease_name = solutions_db[full_class_name]["disease_name"]
            except KeyError:
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

# --- 4. Web App Flask Routes ---
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

# --- NEW: Route for the interface selection page ---
@app.route('/interface', methods=['GET'])
def interface_selection():
    return render_template('interface.html')

@app.route('/app', methods=['GET'])
def launch_app():
    sorted_crops = sorted(crop_list, key=lambda x: x['name'])
    return render_template('app.html', crops=sorted_crops)


@app.route('/predict', methods=['POST'])
def handle_prediction():
    if 'file' not in request.files: return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    crop_key = request.form.get('crop_key')
    if file.filename == '': return jsonify({"error": "No selected file"}), 400
    if not crop_key: return jsonify({"error": "No crop selected"}), 400
    if file:
        # Save the uploaded file permanently
        filename = f"{uuid.uuid4()}_{file.filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        with open(filepath, 'rb') as f:
            image_bytes = f.read()
        
        predictions = predict_image(image_bytes, crop_key)
        
        if predictions:
            return jsonify({
                "predictions": predictions,
                "image_url": url_for('send_uploaded_file', filename=filename) # Return the *new* unique URL
            })
        else:
            return jsonify({"error": "Could not process image"}), 500

@app.route('/solution', methods=['POST'])
def get_solution():
    class_name = request.json.get('class_name')
    if not class_name: return jsonify({"error": "Missing class name"}), 400
    solution = solutions_db.get(class_name) 
    if solution: return jsonify(solution)
    else:
        for key in solutions_db.keys():
            if key.lower() == class_name.lower():
                return jsonify(solutions_db[key])
        return jsonify({"error": f"Solution not found for class: {class_name}"}), 404

@app.route('/uploads/<filename>')
def send_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


# ===================================================================
# --- 5. NEW: ADMIN PANEL ROUTES ---
# ===================================================================

@app.route('/admin', methods=['GET'])
def admin_panel():
    """Serves the main admin review page."""
    # Pass the crop_list to the admin page for its dropdown
    sorted_crops = sorted(crop_list, key=lambda x: x['name'])
    return render_template('admin.html', crops=sorted_crops)

@app.route('/api/submit_review', methods=['POST'])
def submit_for_review():
    """API endpoint for users to submit an image for admin review."""
    data = request.get_json()
    image_url = data.get('image_url')
    ai_guess = data.get('ai_guess')
    
    if not image_url or not ai_guess:
        return jsonify({"error": "Missing image_url or ai_guess"}), 400

    try:
        # Load the current review database
        with open(ADMIN_DB_PATH, 'r') as f:
            review_queue = json.load(f)
        
        # Create a new review item
        new_review = {
            "id": str(uuid.uuid4()),
            "image_url": image_url,
            "ai_guess": ai_guess,
            "status": "pending",
            "corrected_crop": None,
            "corrected_disease": None
        }
        
        review_queue.append(new_review)
        
        # Save the updated database
        with open(ADMIN_DB_PATH, 'w') as f:
            json.dump(review_queue, f, indent=2)
            
        return jsonify({"success": True, "message": "Submitted for review."})

    except Exception as e:
        print(f"Error submitting review: {e}")
        return jsonify({"error": "Could not submit review."}), 500

@app.route('/api/get_review_queue', methods=['GET'])
def get_review_queue():
    """API endpoint for the admin page to fetch all pending reviews."""
    try:
        with open(ADMIN_DB_PATH, 'r') as f:
            review_queue = json.load(f)
        
        pending_reviews = [item for item in review_queue if item['status'] == 'pending']
        return jsonify(pending_reviews)
        
    except Exception as e:
        print(f"Error getting review queue: {e}")
        return jsonify({"error": "Could not load review queue."}), 500

@app.route('/api/submit_correction', methods=['POST'])
def submit_correction():
    """API endpoint for admin to submit their corrected label."""
    data = request.get_json()
    review_id = data.get('id')
    corrected_crop = data.get('crop')
    corrected_disease = data.get('disease')
    
    if not review_id or not corrected_crop or not corrected_disease:
        return jsonify({"error": "Missing data"}), 400
        
    image_url = ""

    try:
        # 1. Update the JSON database
        with open(ADMIN_DB_PATH, 'r') as f:
            review_queue = json.load(f)
        
        item_found = False
        for item in review_queue:
            if item['id'] == review_id:
                item['status'] = "reviewed"
                item['corrected_crop'] = corrected_crop
                item['corrected_disease'] = corrected_disease
                image_url = item['image_url'] # Get the image URL for the CSV
                item_found = True
                break
        
        if not item_found:
            return jsonify({"error": "Review ID not found"}), 404
            
        with open(ADMIN_DB_PATH, 'w') as f:
            json.dump(review_queue, f, indent=2)
            
        # 2. Append to the CSV for future training
        with open(CORRECTED_CSV_PATH, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([image_url, corrected_crop, corrected_disease])
            
        return jsonify({"success": True, "message": "Correction saved."})
        
    except Exception as e:
        print(f"Error submitting correction: {e}")
        return jsonify({"error": "Could not save correction."}), 500

# ===================================================================
# --- 6. WHATSAPP BOT LOGIC (Unchanged) ---
# ===================================================================

user_states = {}
WELCOME_IMAGE_URL = "https://i.postimg.cc/0jsxCffT/Agri-Sentinel.png"
RESULTS_IMAGE_URL = "https://i.postimg.cc/0jsxCffT/Agri-Sentinel.png"
FRUIT_KEYS = ["banana", "litchi", "mangoo", "papaya"]

fruit_crops = sorted([c for c in crop_list if c["key"] in FRUIT_KEYS], key=lambda x: x['name'])
veg_crops = sorted([c for c in crop_list if c["key"] not in FRUIT_KEYS], key=lambda x: x['name'])

def get_crop_batches(crop_list):
    batches = []
    for i in range(0, len(crop_list), 2):
        batches.append(crop_list[i:i+2])
    return batches

fruit_batches = get_crop_batches(fruit_crops)
veg_batches = get_crop_batches(veg_crops)

def send_whatsapp_message(payload):
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json",
    }
    print(f"Sending payload: {json.dumps(payload, indent=2)}")
    response = requests.post(WHATSAPP_API_URL, json=payload, headers=headers)
    response.raise_for_status()
    return response.json()

def send_text_message(to, text):
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": text}
    }
    send_whatsapp_message(payload)

def send_quick_reply(to, text, buttons, header_image_url=None):
    actions = []
    for i, title in enumerate(buttons):
        actions.append({
            "type": "reply",
            "reply": { "id": f"btn_{i}", "title": title }
        })
    
    interactive_payload = {
        "type": "button",
        "body": {"text": text},
        "action": {"buttons": actions}
    }
    
    if header_image_url:
        interactive_payload["header"] = {
            "type": "image",
            "image": {"link": header_image_url}
        }
        
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "interactive",
        "interactive": interactive_payload
    }
    send_whatsapp_message(payload)

def get_media_url(media_id):
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}"}
    response = requests.get(f"https://graph.facebook.com/v19.0/{media_id}", headers=headers)
    response.raise_for_status()
    return response.json().get("url")

def download_image(media_url):
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}"}
    response = requests.get(media_url, headers=headers)
    response.raise_for_status()
    return response.content

def format_solution(solution_data, solution_type):
    if not solution_data: return "I'm sorry, I don't have a solution for that disease."
    if solution_type == "organic":
        icon, data, title = "🌿", solution_data.get("organic_solution"), "Organic Solution"
    elif solution_type == "chemical":
        icon, data, title = "🧪", solution_data.get("chemical_solution"), "Chemical Solution"
    else:
        icon, data, title = "🛡️", {"tips": solution_data.get("prevention_tips", [])}, "Prevention Tips"
    if not data: return f"No {title} information found."
    message = f"{icon} *{title}*\n\n"
    if "strategy" in data: message += f"*Strategy:*\n{data['strategy']}\n\n"
    if "products" in data:
        message += "*Products & Methods:*\n"
        for prod in data["products"]: message += f"- *{prod['name']}:* {prod['details']}\n"
    if "tips" in data:
        for tip in data["tips"]: message += f"- {tip}\n"
    return message.strip()

@app.route("/webhook", methods=["GET", "POST"])
def webhook():
    if request.method == "GET":
        mode = request.args.get("hub.mode")
        token = request.args.get("hub.verify_token")
        challenge = request.args.get("hub.challenge")
        if mode == "subscribe" and token == VERIFY_TOKEN:
            print("Webhook verified.")
            return challenge, 200
        else:
            print("Webhook verification failed.")
            return "Verification failed", 403

    if request.method == "POST":
        try:
            body = request.get_json()
            entry = body.get("entry", [])
            if not entry or not entry[0].get("changes", [])[0].get("value", {}).get("messages"):
                return "OK", 200 
            
            message = body["entry"][0]["changes"][0]["value"]["messages"][0]
            user_id = message["from"]
            message_type = message["type"]
            state_data = user_states.get(user_id, {"state": "START"})
            current_state = state_data.get("state")
            
            if message_type == "text" and message["text"]["body"].lower() == "exit":
                user_states[user_id] = {"state": "START"}
                send_quick_reply(user_id, "OK, let's restart. What would you like to diagnose?", ["Fruit", "Vegetable"])
                return "OK", 200

            if current_state == "START":
                send_quick_reply(user_id, "Welcome to Agri-Sentinel! I can help you diagnose plant diseases.\n\nWhat would you like to diagnose?", ["Fruit", "Vegetable"], header_image_url=WELCOME_IMAGE_URL)
                user_states[user_id] = {"state": "AWAITING_CROP_TYPE"}

            elif current_state == "AWAITING_CROP_TYPE":
                if message_type != "interactive" or message["interactive"]["type"] != "button_reply":
                    send_quick_reply(user_id, "Please select an option.", ["Fruit", "Vegetable"])
                    return "OK", 200
                choice = message["interactive"]["button_reply"]["title"]
                batches = fruit_batches if choice == "Fruit" else veg_batches
                crop_type = "fruit" if choice == "Fruit" else "vegetable"
                buttons = [c["name"] for c in batches[0]]
                if len(batches) > 1: buttons.append("More...")
                send_quick_reply(user_id, f"Please select a {crop_type}:", buttons)
                user_states[user_id] = {"state": "AWAITING_CROP_NAME", "type": crop_type, "batch_index": 0}

            elif current_state == "AWAITING_CROP_NAME":
                if message_type != "interactive" or message["interactive"]["type"] != "button_reply":
                    send_text_message(user_id, "Please select a crop from the buttons.")
                    return "OK", 200
                choice = message["interactive"]["button_reply"]["title"]
                batches = fruit_batches if state_data["type"] == "fruit" else veg_batches
                
                if choice == "More...":
                    next_batch_index = state_data.get("batch_index", 0) + 1
                    if next_batch_index < len(batches):
                        buttons = [c["name"] for c in batches[next_batch_index]]
                        if (next_batch_index + 1) < len(batches): buttons.append("More...")
                        send_quick_reply(user_id, "Please select a crop:", buttons)
                        user_states[user_id]["batch_index"] = next_batch_index
                    else:
                        send_text_message(user_id, "No more crops to show.")
                else:
                    selected_crop = next((c for c in crop_list if c["name"] == choice), None)
                    if selected_crop:
                        send_text_message(user_id, f"Great! You selected *{selected_crop['name']}*.\n\nPlease upload a clear photo of the affected plant or fruit.")
                        user_states[user_id] = {"state": "AWAITING_IMAGE", "crop_key": selected_crop["key"]}
                    else:
                        send_text_message(user_id, "I'm sorry, I didn't recognize that crop. Let's try again.")
                        user_states[user_id] = {"state": "START"}
                        send_quick_reply(user_id, "What would you like to diagnose?", ["Fruit", "Vegetable"])

            elif current_state == "AWAITING_IMAGE":
                if message_type != "image":
                    send_text_message(user_id, "Please upload an *image* for diagnosis. You can also send 'exit' to restart.")
                    return "OK", 200
                media_id = message["image"]["id"]
                media_url = get_media_url(media_id)
                image_bytes = download_image(media_url)
                send_text_message(user_id, "Analyzing your image... 🔬")
                predictions = predict_image(image_bytes, state_data["crop_key"])
                if not predictions:
                    send_text_message(user_id, "I'm sorry, I couldn't analyze that image. Please try another one or send 'exit' to restart.")
                    return "OK", 200
                top_1, top_2 = predictions[0], predictions[1]
                solution_data_1 = solutions_db.get(top_1["full_class_name"])
                solution_data_2 = solutions_db.get(top_2["full_class_name"])
                desc_1 = solution_data_1.get("description", "No description available.") if solution_data_1 else "No description available."
                desc_2 = solution_data_2.get("description", "No description available.") if solution_data_2 else "No description available."
                if len(desc_1) > 150: desc_1 = desc_1[:150] + "..."
                if len(desc_2) > 150: desc_2 = desc_2[:150] + "..."
                reply_text = (
                    f"Diagnosis complete. Here are the top 2 results:\n\n"
                    f"1. *{top_1['disease_name']}* (Confidence: {top_1['confidence']}%)\n"
                    f"_{desc_1}_\n\n"
                    f"2. *{top_2['disease_name']}* (Confidence: {top_2['confidence']}%)\n"
                    f"_{desc_2}_\n\n"
                    f"Please select a solution to explore for *{top_1['disease_name']}*. You can also send 'exit' to restart."
                )
                send_quick_reply(user_id, reply_text, ["Organic Solution", "Chemical Solution", "Preventive Measures"], header_image_url=RESULTS_IMAGE_URL)
                user_states[user_id] = {"state": "IN_SOLUTION_LOOP", "top_class": top_1["full_class_name"]}

            elif current_state == "IN_SOLUTION_LOOP":
                if message_type != "interactive" or message["interactive"]["type"] != "button_reply":
                    send_quick_reply(user_id, "Please choose a solution type or send 'exit' to restart.", ["Organic Solution", "Chemical Solution", "Preventive Measures"])
                    return "OK", 200
                choice = message["interactive"]["button_reply"]["title"]
                top_class = state_data["top_class"]
                solution_data = solutions_db.get(top_class)
                solution_message = ""
                if choice == "Organic Solution": solution_message = format_solution(solution_data, "organic")
                elif choice == "Chemical Solution": solution_message = format_solution(solution_data, "chemical")
                elif choice == "Preventive Measures": solution_message = format_solution(solution_data, "prevention")
                send_text_message(user_id, solution_message)
                send_quick_reply(user_id, "What would you like to see next? Send 'exit' to restart.", ["Organic Solution", "Chemical Solution", "Preventive Measures"])

        except Exception as e:
            print(f"Error handling webhook POST: {e}")
        return "OK", 200

# --- 7. Run the App ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)