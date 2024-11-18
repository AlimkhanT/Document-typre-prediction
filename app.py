from flask import Flask, request, jsonify, render_template
import torch
from torchvision import transforms, models
from PIL import Image
import os

app = Flask(__name__)

# Load Models
class_names = ['specification', 'email', 'scientific_report', 'budget', 'scientific_publication', 'resume']
num_classes = 6 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Instantiate the baseline model
baseline_model = models.resnet50(weights=None)  # Create the model without pretrained weights
baseline_model.fc = torch.nn.Linear(baseline_model.fc.in_features, num_classes)  # Modify the final layer

# Load the baseline model's state dict (weights)
checkpoint = torch.load('/Users/alimkhan/Downloads/capstone-main/aml_final/backend/baseline_model.pth', map_location=device)
baseline_model.load_state_dict(checkpoint)

# Switch to evaluation mode
baseline_model.eval()

# Instantiate the enhanced model
enhanced_model = models.resnet50(weights=None)  # Create ResNet-50 without pretrained weights

# Modify the final layers of the Enhanced Model
enhanced_model.fc = torch.nn.Sequential(
    torch.nn.BatchNorm1d(enhanced_model.fc.in_features),
    torch.nn.Linear(enhanced_model.fc.in_features, 512),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.5),
    torch.nn.Linear(512, num_classes)
)

# Load the Enhanced Model's weights
enhanced_checkpoint = torch.load('/Users/alimkhan/Downloads/capstone-main/aml_final/backend/enhanced_model.pth', map_location=device)
enhanced_model.load_state_dict(enhanced_checkpoint)

# Set the Enhanced Model to evaluation mode
enhanced_model.eval()

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.route('/')
def home():
    return render_template('index.html')  # Make sure this HTML file exists

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Check if the file is a .jpg
    if not file.filename.lower().endswith('.jpg'):
        return jsonify({"error": "Invalid file type. Only .jpg files are allowed."}), 400

    try:
        # Open and preprocess the image
        image = Image.open(file).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)

        # Make prediction with baseline model
        with torch.no_grad():
            baseline_outputs = baseline_model(input_tensor)
            _, baseline_predicted = torch.max(baseline_outputs, 1)
            baseline_label = class_names[baseline_predicted.item()]

        # Make prediction with enhanced model
        with torch.no_grad():
            enhanced_outputs = enhanced_model(input_tensor)
            _, enhanced_predicted = torch.max(enhanced_outputs, 1)
            enhanced_label = class_names[enhanced_predicted.item()]

        return jsonify({
            "baseline_result": baseline_label,
            "enhanced_result": enhanced_label
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
