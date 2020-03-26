import io
import numpy as np

from flask import Flask, jsonify, request
from PIL import Image
app = Flask(__name__)

@app.route('/style/api/v1.0/segmentate', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img = Image.open(io.BytesIO(file.read()))
        
        return 