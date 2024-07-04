#Lablab.ai Next-create-startup-Hackthon Submission.

# Diet Vision

## Overview
This project involves building a system to automatically track and predict the nutritional values of food items from images. The system uses computer vision and machine learning models deployed on an inference server to analyze images and return nutritional information.

## Tech Stack

### Hardware
- **Armband with Camera:** Used to capture images of food items.
- **Continuous Glucose Monitor (CGM):** Measures glucose levels continuously.

### Software

#### Model
- **Model Architecture:** InceptionV3, will be Updating to YOLOV8 .
- **Model Training Framework:** TensorFlow (Will be moving on to pytorch)
- **Dataset Used:** Nutrition5k by Google
  - **Max Values Calculation:** The max values for normalization (used for de-normalizing the output) are calculated from the Nutrition5k dataset by taking the maximum value of each nutritional attribute (calories, mass, fat, carbohydrates, protein) from the dataset.

#### Model Serving
- **Inference Server:** Triton Inference Server
  - **Version:** 2.21.0
  - **Deployment:** Docker

#### Backend API
- **Framework:** Flask
- **Others:** Opencv, numpy, pandas
- **Language:** Python
- **API Endpoint:** `/predict`
- **URL:** `https://<server-ip>:5000/predict`

## Installation and Setup

### 1. Triton Inference Server

- Pull the Triton Inference Server Docker image and run it:

  ```sh
  docker run --gpus all --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 -v <model-repo-path>:/models nvcr.io/nvidia/tritonserver:21.04-py3 tritonserver --model-repository=/models
  ```

### 2. Flask API

- Install required Python packages:

  ```sh
  pip install flask numpy opencv-python-headless tritonclient
  ```

- Create the Flask API (`app.py`):

- Run the Flask API:

  ```sh
  python app.py
  ```

## Usage

1. **Start the Triton Inference Server:**

   ```sh
   docker run --gpus all --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 -v <model-repo-path>:/models nvcr.io/nvidia/tritonserver:[XX.XX]-py3 tritonserver --model-repository=/models
   ```

2. **Start the Flask API:**

   ```sh
   python app.py
   ```

3. **Send a POST request to the API endpoint:**

   - URL: `https://<server-ip>:5000/predict`
   - Method: POST
   - Form Data: Key `image`, Value: Image file

   ```sh
   curl -X POST -F 'image=@path_to_image.jpg' https://<server-ip>:5000/predict
   ```

## Notes

- Ensure that the Triton Inference Server and Flask API are running on the same network and that the ports are correctly mapped.
- The max values for de-normalizing the output were calculated from the Nutrition5k dataset by taking the maximum value of each nutritional attribute.
