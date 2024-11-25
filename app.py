from flask import Flask, request, jsonify, render_template, send_file
import torch
from torchvision import models, transforms
from PIL import Image
from fpdf import FPDF
import os
import csv
from torch import nn

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Class names and prices
class_names = ['burger', 'butter_naan', 'chai', 'chapati', 'chole_bhature', 
               'dal_makhani', 'dhokla', 'fried_rice', 'idli', 'jalebi', 
               'kaathi_rolls', 'kadai_paneer', 'kulfi', 'masala_dosa', 
               'momos', 'paani_puri', 'pakode', 'pav_bhaji', 'pizza', 'samosa']

price_dict = {}
with open(r'food_prices.csv', mode='r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        price_dict[row['food']] = float(row['price'])

# Model setup
model_path = r'best_model.pth'
model = models.efficientnet_b3(pretrained=True)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(class_names))
model.load_state_dict(torch.load(model_path))
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

table_bills = {}

def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return class_names[predicted.item()]

def generate_pdf_bill(table_number):
    items = table_bills.get(table_number, [])
    if not items:
        return None

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 12)
    pdf.cell(200, 10, f"Bill for Table {table_number}", ln=True, align="C")
    pdf.ln(10)
    pdf.cell(100, 10, "Item", border=1)
    pdf.cell(40, 10, "Price", border=1, ln=True)

    total = 0
    for item_name, price in items:
        pdf.cell(100, 10, item_name, border=1)
        pdf.cell(40, 10, f"{price:.2f}", border=1, ln=True)
        total += price

    pdf.ln(10)
    pdf.cell(100, 10, "Total", border=1)
    pdf.cell(40, 10, f"{total:.2f}", border=1, ln=True)

    pdf_file = os.path.join(UPLOAD_FOLDER, f"Table_{table_number}_bill.pdf")
    pdf.output(pdf_file)
    return pdf_file

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    table_number = request.form['table_number']
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    predicted_class = predict_image(filepath)
    price = price_dict.get(predicted_class, 0)
    
    if table_number not in table_bills:
        table_bills[table_number] = []
    table_bills[table_number].append((predicted_class, price))
    
    return jsonify({'item': predicted_class, 'price': price})

@app.route('/generate_bill/<table_number>', methods=['GET'])
def generate_bill(table_number):
    pdf_path = generate_pdf_bill(table_number)
    if pdf_path:
        return send_file(pdf_path, as_attachment=True)
    return "No items to bill."

if __name__ == '__main__':
    app.run(debug=True)