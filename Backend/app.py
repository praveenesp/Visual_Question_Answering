from flask import Flask, request, jsonify
import os
from vqa import *
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Define a route to handle image uploads
@app.route('/predict', methods=['POST'])
def predict():
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'Empty image file provided'}), 400
    
    # Save the image temporarily
    image_path = 'temp_image.jpg'
    image_file.save(image_path)

    text = request.form.get('text')
    # Perform prediction with VQA
    result = Vqa(image_path, text)
    
    # Remove the temporary image file
    os.remove(image_path)
    
    return jsonify({'prediction': result}), 200


if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=5000)

