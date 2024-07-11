#!/bin/bash

# Download the ZIP file from the provided URL
curl -L "https://universe.roboflow.com/ds/lNcS7jChle?key=vD29aNt24F" -o roboflow.zip

# Unzip the downloaded file
unzip roboflow.zip

# Remove the ZIP file to clean up
rm roboflow.zip

mkdir -p sketch_images

mv train test valid README.dataset.txt README.roboflow.txt data.yaml sketch_images/


