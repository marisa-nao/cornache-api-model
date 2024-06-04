import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array

def preprocess_image_as_array(image_file, target_size=(256, 256)):
    im = Image.open(image_file).convert('RGB')
    im = im.resize(target_size)
    img_array = img_to_array(im)
    img_array = img_array[..., ::-1]  # Convert RGB to BGR
    img_array = img_array.astype('float') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array