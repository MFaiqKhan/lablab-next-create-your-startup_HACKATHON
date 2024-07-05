# Lablab.ai Next-create-startup-Hackathon Submission

## Diet Vision

### Overview

Diet Vision is a pioneering system designed to automatically track and predict the nutritional values of food items from images. By leveraging advanced computer vision and machine learning models, our solution analyzes food images and provides comprehensive nutritional information, enhancing dietary tracking and health management.

### Tech Stack

#### Hardware
- **Armband with Camera:** Captures images of food items in real-time.
- **Continuous Glucose Monitor (CGM):** Continuously measures glucose levels, providing critical data for dietary analysis.

#### Software

**Model**
- **Architecture:** Initially using InceptionV3, transitioning to YOLOV8 for improved accuracy and efficiency.
- **Framework:** TensorFlow for current model training, with plans to migrate to PyTorch.
- **Dataset:** Utilizes the Nutrition5k dataset by Google to train and validate the models. The maximum values for normalization, used for de-normalizing the output, are derived from this dataset.

**Model Serving**
- **Inference Server:** NVIDIA Triton Inference Server deployed via AWS SageMaker. This setup ensures robust and scalable model serving capabilities.

**Backend API**
- **Framework:** Flask, providing a lightweight and flexible API backend.
- **Additional Libraries:** OpenCV for image processing, NumPy for numerical operations, and Pandas for data manipulation, ONNX, ONNXRuntime for Backend Model, Nvidia Triton Inference , Sagemaker.
- **Language:** Python
- **Endpoint:** Deployed to EC2 AWS`, contact us to get the endpoint directly

**Deployment**
- **Cloud Services:** 
  - **AWS SageMaker:** For deploying the Triton Inference Server.
  - **EC2:** Hosting the Flask API with NGINX acting as a reverse proxy to manage traffic.

### Features

1. **Automated Nutritional Analysis:**
   - Users can take photos of their meals using an armband camera.
   - The system processes these images to provide detailed nutritional breakdowns, including calories, mass, fat, carbohydrates, and protein content.

2. **Real-time Glucose Monitoring:**
   - Integration with a Continuous Glucose Monitor allows users to track their glucose levels in real time, enabling personalized dietary recommendations.

3. **User-Friendly Interface:**
   - A web application, currently under development, will offer user account management and personalized dietary tracking.
   - A Streamlit-based web app and a mobile application are also in the pipeline to enhance accessibility and user experience.

### Visuals

Here are some sample images analyzed by Diet Vision, showcasing the predicted nutritional values:

![Pizza](images\Image1.png)
![A Plate full of good Meal](images\Images2.png)

### Future Developments

- **Web Application:** We are developing a full-fledged web app to manage user accounts, track dietary intake, and offer personalized nutrition advice.
- **Streamlit and Mobile Apps:** Streamlit will power our interactive web app, while a mobile app will provide users with on-the-go access to our services.
- **Enhanced Model Accuracy:** By transitioning to YOLOV8 and PyTorch, we aim to further refine our model's accuracy and efficiency.

**Disclaimer** : It's still under development so Accuracy can be off. We are Continuously working ...

**Join us in making a healthier future a reality with Diet Vision.**