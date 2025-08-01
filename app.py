from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from segmentation_models import Unet
from segmentation_models.losses import bce_dice_loss
from segmentation_models.metrics import iou_score, f1_score
from segmentation_models.base.objects import SumOfLosses  # ✅ This is key!
import os
import uuid
import tempfile
import shutil
import matplotlib.pyplot as plt

# Important: this must come BEFORE load_model
# It registers all custom objects used internally (e.g., FixedDropout, swish, etc.)
import segmentation_models as sm
sm.set_framework('tf.keras')
sm.framework()

model = tf.keras.models.load_model(
    r"G:\Projects\Cellula_Internship\Water_Bodies_Semantic_Segmentation\best_pretrained_model.keras",
    compile=False
)

# Initialize Flask app
app = Flask(__name__)

SELECTED_BANDS = [4, 5, 11]

def normalize_bandwise(img):
    normalized = np.zeros_like(img, dtype=np.float32)
    for band in range(img.shape[-1]):
        band_data = img[..., band]
        band_min = np.min(band_data)
        band_max = np.max(band_data)
        if band_max - band_min == 0:
            normalized[..., band] = 0.0
        else:
            normalized[..., band] = (band_data - band_min) / (band_max - band_min)
    return normalized

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        image = np.array(data["image"], dtype=np.float32)

        if image.shape[-1] != 12:
            return jsonify({"error": "Expected 12 bands."}), 400

        image = normalize_bandwise(image)
        image = image[..., SELECTED_BANDS]
        image = np.expand_dims(image, axis=0)

        prediction = model.predict(image)[0]
        predicted_mask = (prediction > 0.5).astype(np.uint8).squeeze().tolist()

        # Generate a unique filename
        filename = f"pred_{uuid.uuid4().hex[:8]}.png"
        temp_dir = tempfile.gettempdir()
        save_path = os.path.join(temp_dir, filename)

        # Save the predicted mask as an image
        plt.imsave(save_path, predicted_mask, cmap='gray')

        # Return the path or image link (for testing only)
        return jsonify({
            "prediction_path": save_path,
            "note": "Temporary prediction image saved. File will be deleted by OS cleanup later."
        })


        return jsonify({"prediction": predicted_mask})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
