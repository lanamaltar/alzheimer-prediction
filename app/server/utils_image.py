# utils_image.py

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from tensorflow.keras.preprocessing import image
import numpy as np

def preprocess_image(img_path, target_size=(128, 128)):
    """
    Učitaj sliku, promijeni veličinu, pretvori u numpy i normaliziraj
    """
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normalizacija
    img_array = np.expand_dims(img_array, axis=0)  # Oblik: (1, 128, 128, 3)
    return img_array

