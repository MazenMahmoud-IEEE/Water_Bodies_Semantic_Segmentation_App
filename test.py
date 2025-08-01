import requests
import numpy as np
import json
import requests
import numpy as np
import tifffile
import os
import uuid
import tempfile
import shutil
import matplotlib.pyplot as plt

# Load the .tif image using tifffile
image_path = r"G:\Projects\Cellula_Internship\Water_Bodies_Semantic_Segmentation\data\images\23.tif"
image = tifffile.imread(image_path).astype(np.float32)

# Convert image to list format (serializable)
payload = {"image": image.tolist()}

# Send POST request
response = requests.post("http://127.0.0.1:5000/predict", json=payload)

# Print prediction result
print("Prediction received:", response.json())

