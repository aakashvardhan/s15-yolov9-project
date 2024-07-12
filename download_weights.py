import os
import requests

# Define the home directory and weights directory
home_dir = os.path.expanduser("/home/ubuntu/s15-yolov9-project")
weights_dir = os.path.join(home_dir, "weights")

# Create the weights directory if it doesn't exist
os.makedirs(weights_dir, exist_ok=True)

# Define the URLs for the YOLOv9 model files
urls = [
    "https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-c.pt",
    "https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-e.pt"
]

# Download the YOLOv9 model files
for url in urls:
    response = requests.get(url)
    file_name = os.path.join(weights_dir, os.path.basename(url))
    with open(file_name, 'wb') as file:
        file.write(response.content)