import requests

url = 'http://34.203.252.91' # Deployed on EC2
image_path = 'rgb.png'  # Adjusted path separator for Windows

# Read the image file
with open(image_path, 'rb') as img_file:
    files = {'image': img_file}
    response = requests.post(url, files=files, verify=False)  # Set verify=True if you have a valid SSL certificate

# Print the response status code and text
print(f"Response Status Code: {response.status_code}")
print(f"Response Text: {response.text}")

# Attempt to print the response as JSON
try:
    print(response.json())
except ValueError as e:
    print(f"Failed to parse JSON: {e}")
