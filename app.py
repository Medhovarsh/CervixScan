from flask import Flask, render_template, request, redirect, url_for, flash
import os
from werkzeug.utils import secure_filename
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import base64

# ====================== Flask App Setup ======================
app = Flask(__name__)
app.secret_key = "cervical_cancer_prediction_secret_key"

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ====================== Define Model ======================
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 56 * 56, 128),  # ⚠️ Change if needed
            nn.ReLU(),
            nn.Linear(128, 3)  # 3 classes
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

# ====================== Load PyTorch Model ======================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNNModel()
model.load_state_dict(torch.load("converted_model.pth", map_location=device))
model.eval()

# ====================== Helper Functions ======================
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    """Prepare image for PyTorch model"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)  # Add batch dim
    return img_tensor.to(device)

def get_prediction(image_path):
    """Get class prediction from model"""
    classes = ["Normal", "Precancerous", "Cancerous"]
    input_tensor = preprocess_image(image_path)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)
        return classes[predicted.item()], confidence.item()

# ====================== Routes ======================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            prediction, confidence = get_prediction(filepath)

            with open(filepath, "rb") as img_file:
                img_data = base64.b64encode(img_file.read()).decode('utf-8')

            return render_template('result.html',
                                   prediction=prediction,
                                   confidence=round(confidence * 100, 2),
                                   image_data=img_data)

    return render_template('predict.html')

# ====================== Run ======================
if __name__ == '__main__':
    app.run(debug=True)
