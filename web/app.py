import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import io
import requests
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Function to get data from API
def get_meal_data(image):
    url = 'http://34.203.252.91/'
    try:
        with io.BytesIO() as img_file:
            image.save(img_file, format='PNG')
            img_file.seek(0)
            files = {'image': img_file}
            response = requests.post(url, files=files, verify=False)  # Set verify=True if you have a valid SSL certificate
        
        if response.status_code == 200:
            logger.info("Successfully received data from API")
            return response.json()
        else:
            logger.error(f"API request failed with status code: {response.status_code}")
            return {}
    except Exception as e:
        logger.error(f"Error in get_meal_data: {str(e)}")
        return {}


def predict_nutrition(image):
    if image is None:
        raise ValueError("No image provided")
    
    nutrition_data = get_meal_data(image)
    
    if not nutrition_data:
        logger.warning("No nutrition data received from API")
    else:
        logger.info("Successfully predicted nutrition data")
    
    return nutrition_data

# Set page config
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

            adjusted_nutrition = {k: v * consumed_percentage / 100 for k, v in nutrition_data.items()}

            st.header("Adjusted Nutrition")
            col1, col2, col3 = st.columns(3)
            col1.metric("Adjusted Calories", f"{adjusted_nutrition.get('calories', 0):.0f} cal")
            col2.metric("Adjusted Protein", f"{adjusted_nutrition.get('protein', 0):.1f} g")
            col3.metric("Adjusted Carbs", f"{abs(adjusted_nutrition.get('carbohydrates', 0)):.1f} g")
        else:
            st.error("Failed to retrieve nutrition data from the API. Please try again or contact support.")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        st.error(f"An unexpected error occurred: {str(e)}. Please try again or contact support.")
else:
    st.info("Please upload a meal photo to start tracking your nutrition.")
