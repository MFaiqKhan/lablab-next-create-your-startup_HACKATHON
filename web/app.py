import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import datetime
import requests

# Load the model with error handling
try:
    model = tf.keras.models.load_model('raw_models/inceptionv3_tf.h5')
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Function to get data from API
def get_meal_data(image):
    url = 'http://34.203.252.91/'
    
    with io.BytesIO() as img_file:
        image.save(img_file, format='PNG')
        img_file.seek(0)
        files = {'image': img_file}
        try:
            response = requests.post(url, files=files, verify=False)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"API request failed: {e}")
            return {}

# Function to predict nutrition from image
def predict_nutrition(image):
    try:
        img = image.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        prediction = model.predict(img_array)
        
        nutrition_data = get_meal_data(image)
        return nutrition_data
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return {}

st.set_page_config(
    page_title="Diet Vision 🌱",
    page_icon="🥗",
    layout="centered",
    initial_sidebar_state="auto"
)
st.title("Diet Vision")
st.markdown("***A Diet Vision For a Healthier Tomorrow.***")

uploaded_file = st.file_uploader("Choose a meal photo", type=["jpg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Meal Photo", use_column_width=True)
        
        nutrition_data = predict_nutrition(image)
        
        if nutrition_data:
            st.header("Nutrition Information")
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Mass", f"{nutrition_data.get('mass', 0):.0f} g")
            col2.metric("Calories", f"{nutrition_data.get('calories', 0):.1f} c")
            col3.metric("Protein", f"{nutrition_data.get('protein', 0):.1f} g")
            col4.metric("Carbs", f"{abs(nutrition_data.get('carbohydrates', 0)):.1f} g")
            col5.metric("Fat", f"{nutrition_data.get('fat', 0):.1f} g")
            
            st.header("Daily Nutrition Goals")
            calories_progress = nutrition_data.get('calories', 0) / 2000  # Assuming 2000 kcal daily goal
            st.progress(calories_progress)
            st.text(f"{calories_progress*100:.1f}% of daily calorie goal")
            
            st.header("Nutrients")
            for nutrient, amount in nutrition_data.items():
                st.text(f"{nutrient.capitalize()}: {abs(amount):.1f}")

            
            st.header("Adjust Consumed Amount")
            consumed_percentage = st.slider("Percentage of meal consumed", 0, 100, 100)
            
            # Update nutrition
            adjusted_nutrition = {k: v * consumed_percentage / 100 for k, v in nutrition_data.items()}
            
            st.header("Adjusted Nutrition")
            col1, col2, col3 = st.columns(3)
            col1.metric("Adjusted Calories", f"{adjusted_nutrition.get('calories', 0):.0f} cal")
            col2.metric("Adjusted Protein", f"{adjusted_nutrition.get('protein', 0):.1f} g")
            col3.metric("Adjusted Carbs", f"{abs(adjusted_nutrition.get('carbohydrates', 0)):.1f} g")

        else:
            st.error("Failed to retrieve nutrition data from the API.")
    except Exception as e:
        st.error(f"Error processing uploaded file: {e}")
else:
    st.info("Please upload a meal photo to start tracking your nutrition.")