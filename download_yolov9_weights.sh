#!/bin/bash

# Define the HOME variable if not already set
: ${HOME:=$HOME}

# Create the weights directory if it doesn't exist
mkdir -p "${HOME}/weights"

# Download the YOLOv9 model files
wget -P "${HOME}/weights" -q https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-c.pt
wget -P "${HOME}/weights" -q https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-e.pt
