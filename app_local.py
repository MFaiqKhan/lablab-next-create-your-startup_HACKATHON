from flask import Flask, request, jsonify
import numpy as np
import cv2
import tritonclient.http as httpclient

app = Flask(__name__)

# Define model parameters
model_input_name = "input"
model_output_names = ["total_calories_neuron", "total_carb_neuron", "total_fat_neuron", "total_mass_neuron", "total_protein_neuron"]
model_name = "inceptionv3"
model_vers = "1"
server_url = "localhost:8000"

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
    client = httpclient.InferenceServerClient(url=server_url)
    
    input_img = preprocess_image(image)
    
    inputs = httpclient.InferInput(model_input_name, input_img.shape, "FP32")
    inputs.set_data_from_numpy(input_img)

    outputs = [httpclient.InferRequestedOutput(name) for name in model_output_names]
    
    results = client.infer(model_name, model_version=model_vers, inputs=[inputs], outputs=outputs)

    # Collect and flatten output data
    output_data = []
    for name in model_output_names:
        data = results.as_numpy(name)
        print(f"Output '{name}' shape: {data.shape}")
        output_data.append(data.flatten())

    output_data = np.concatenate(output_data)
    max_values = np.array([9485.81543, 7975, 875.5410156, 844.5686035, 312.491821])
    predicted_values = output_data * max_values

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
    app.run(host='0.0.0.0', port=5000)
