import os
import tensorflow as tf

# Use os.path to construct the model path
model_dir = "model"
model_filename = "my_model.h5"
model_path = os.path.join(model_dir, model_filename)
print(model_path)

# Load the model directly from the local path
try:
    best_model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully from", model_path)
except Exception as e:
    print("Error loading model:", e)
    best_model = None  # Handle the error case (optional)

# List of class names
class_names = ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']
