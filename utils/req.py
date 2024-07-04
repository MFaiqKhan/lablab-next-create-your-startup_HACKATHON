import requests

url = 'http://localhost:5000/predict'
image_path = 'test_images\gnl5o4tzh7ad1.jpeg'

# Read the image file
with open(image_path, 'rb') as img_file:
    files = {'image': img_file}
    response = requests.post(url, files=files, verify=False)  # Set verify=True if you have a valid SSL certificate

# Print the response
print(response.json())
