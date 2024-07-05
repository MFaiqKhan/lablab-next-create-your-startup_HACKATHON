from flask import Flask, request, jsonify
import numpy as np
import cv2
import boto3
import json
from dotenv import load_dotenv
import os

load_dotenv()
endpoint_name = os.getenv("ENDPOINT_NAME")

app = Flask(__name__)

# Define model parameters
endpoint_name = endpoint_name 
runtime_sm_client = boto3.client("sagemaker-runtime")

# Preprocess the image
def preprocess_image(image):
    img = cv2.imdecode(np.frombuffer(image, np.uint8), -1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))  
    img = img / 255.0
    img = np.expand_dims(img, axis=0).astype(np.float32)
    return img

# Predict nutritional values
def predict_nutrition(image):
    input_img = preprocess_image(image)
    
    payload = {
        "inputs": [
            {
                "name": "input",
                "shape": input_img.shape,
                "datatype": "FP32",
                "data": input_img.tolist()
            }
        ]
    }
    
    response = runtime_sm_client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Body=json.dumps(payload)
    )
    
    result = json.loads(response['Body'].read().decode('utf8'))
    output_data = [np.array(result['outputs'][i]['data']) for i in range(len(result['outputs']))]

    max_values = np.array([9485.81543, 7975, 875.5410156, 844.5686035, 312.491821])
    predicted_values = np.concatenate(output_data) * max_values

    return {
        "calories": predicted_values[0].item(),
        "mass": predicted_values[1].item(),
        "fat": predicted_values[2].item(),
        "carbohydrates": predicted_values[3].item(),
        "protein": predicted_values[4].item()
    }

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    image = request.files['image'].read()
    predictions = predict_nutrition(image)
    
    return jsonify(predictions)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)



