import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import sys
import os


def load_and_preprocess_image(img_path, target_size=(150, 150)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array


def main(image_path):
    if not os.path.exists(image_path):
        print(f"File {image_path} does not exist.")
        return

    model = load_model('cat_vs_dog_classifier.h5')
    img_array = load_and_preprocess_image(image_path)
    prediction = model.predict(img_array)

    if prediction[0] > 0.5:
        print("The image is a dog.")
    else:
        print("The image is a cat.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_model.py <path_to_image>")
    else:
        main(sys.argv[1])
