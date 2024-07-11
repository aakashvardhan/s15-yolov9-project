import os
import requests
import zipfile

# Define the URL and the output file name
url = "https://universe.roboflow.com/ds/lNcS7jChle?key=vD29aNt24F"
zip_file = "roboflow.zip"

# Download the ZIP file from the provided URL
response = requests.get(url)
with open(zip_file, 'wb') as file:
    file.write(response.content)

# Unzip the downloaded file
with zipfile.ZipFile(zip_file, 'r') as zip_ref:
    zip_ref.extractall()

# Remove the ZIP file to clean up
os.remove(zip_file)

# Create the directory if it doesn't exist
os.makedirs("sketch_images", exist_ok=True)

# Move the files to the new directory
for file_name in ["train", "test", "valid", "README.dataset.txt", "README.roboflow.txt", "data.yaml"]:
    os.rename(file_name, os.path.join("sketch_images", file_name))