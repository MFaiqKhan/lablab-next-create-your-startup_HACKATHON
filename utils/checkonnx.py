import numpy as np
import cv2
import onnxruntime as ort

# Set the image dimensions to match the model's expected input size
IMG_DIMN = 224  # Change as per your model's requirements

# Load the maximum values for denormalization (these values should be known or calculated beforehand)
max_calorie = 9485.81543  
max_mass = 7975  
max_fat = 875.5410156  
max_carb = 844.5686035  
max_protein = 312.491821  

# Preprocess the image
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_DIMN, IMG_DIMN))
    img = img / 255.0
    return img

# Function to predict nutritional values
def predict_nutrition(img_path, session, input_name, output_names):
    preprocessed_img = preprocess_image(img_path)
    preprocessed_reshaped_img = np.expand_dims(preprocessed_img, axis=0).astype(np.float32)
    
    outputs = session.run(output_names, {input_name: preprocessed_reshaped_img})
    
    # Assuming the model's output order is [calories, mass, fat, carb, protein]
    predicted_calories = outputs[0] * max_calorie
    predicted_mass = outputs[1] * max_mass
    predicted_fat = outputs[2] * max_fat
    predicted_carb = outputs[3] * max_carb
    predicted_protein = outputs[4] * max_protein
    
    print('Predicted Nutritional Values:')
    print('Calories: ', predicted_calories)
    print('Mass: ', predicted_mass)
    print('Fat: ', predicted_fat)
    print('Carbohydrates: ', predicted_carb)
    print('Protein: ', predicted_protein)

    # Ensure these are scalar values before formatting
    predicted_calories = predicted_calories.item()  # Convert NumPy scalar to Python scalar
    predicted_mass = predicted_mass.item()
    predicted_fat = predicted_fat.item()
    predicted_carb = predicted_carb.item()
    predicted_protein = predicted_protein.item()

    # Load the original image for display
    img_display = cv2.imread(img_path)

    # Resize the image if it's too big
    max_width = 600  # Define maximum width of the output image
    height, width = img_display.shape[:2]
    
    # Calculate scaling factor while maintaining aspect ratio
    scale_factor = max_width / width
    new_height = int(height * scale_factor)
    
    # Resize the image using cv2.INTER_AREA for shrinking
    img_displayr = cv2.resize(img_display, (max_width, new_height), interpolation=cv2.INTER_AREA)
    
    # Define text properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2

    # Colors for each nutritional value remain the same
    calories_color = (0, 255, 0)  # Green
    mass_color = (255, 127, 127)     # Blue
    fat_color = (0, 0, 255)      # Red
    carb_color = (255, 255, 0)   # Yellow
    protein_color = (0, 255, 255)# Cyan
    
    # Draw nutritional values on the image
    cv2.putText(img_displayr, f'Calories: {predicted_calories:.2f}', (50, 50), font, font_scale, calories_color, thickness)
    cv2.putText(img_displayr, f'Mass: {predicted_mass:.2f}g', (50, 80), font, font_scale, mass_color, thickness)
    cv2.putText(img_displayr, f'Fat: {predicted_fat:.2f}g', (50, 110), font, font_scale, fat_color, thickness)
    cv2.putText(img_displayr, f'Carbs: {predicted_carb:.2f}g', (50, 140), font, font_scale, carb_color, thickness)
    cv2.putText(img_displayr, f'Protein: {predicted_protein:.2f}g', (50, 170), font, font_scale, protein_color, thickness)
    
    # Display the image with nutritional values
    cv2.imshow('Nutritional Values', img_displayr)
    cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows()

# Load the ONNX model
onnx_model_path = 'models\inceptionv3\\1\inceptionv3.onnx'
session = ort.InferenceSession(onnx_model_path)
input_name = session.get_inputs()[0].name
output_names = [output.name for output in session.get_outputs()]

# Example usage
image_path = 'test_images\ezgif-3-b32a018f8b.jpg'
predict_nutrition(image_path, session, input_name, output_names)
