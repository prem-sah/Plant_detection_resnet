import tensorflow as tf
import numpy as np
import json
import os
import argparse
from tensorflow.keras.preprocessing import image

IMG_SIZE = 224
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "plant_disease_model.keras")
CLASS_MAP_PATH = os.path.join(BASE_DIR, "class_indices.json")
DEFAULT_PREDICT_PATH = "/Users/sushilkumarpatel/Desktop/plant_detection_prem/Plant_disease_detection_resne/dataset_test"
model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_MAP_PATH) as f:
    class_indices = json.load(f)

index_to_class = {v:k for k,v in class_indices.items()}


def predict(img_path):

    img = image.load_img(img_path, target_size=(IMG_SIZE,IMG_SIZE))
    img_array = image.img_to_array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)

    index = int(np.argmax(pred))
    confidence = np.max(pred)

    print(f"Image: {img_path}")
    print(f"Prediction: {index_to_class[index]} (Class #{index})")
    print("Confidence:", round(confidence*100,2), "%")
    print("-" * 60)


def predict_path(target_path):
    if os.path.isfile(target_path):
        predict(target_path)
        return

    if not os.path.isdir(target_path):
        raise FileNotFoundError(f"Path does not exist: {target_path}")

    image_files = []
    for root, _, files in os.walk(target_path):
        for file_name in files:
            if file_name.lower().endswith(VALID_EXTENSIONS):
                image_files.append(os.path.join(root, file_name))

    if not image_files:
        print(f"No image files found in: {target_path}")
        return

    print(f"Found {len(image_files)} images. Running predictions...\n")
    for img_file in sorted(image_files):
        predict(img_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict plant disease from image or folder")
    parser.add_argument(
        "--path",
        type=str,
        default=DEFAULT_PREDICT_PATH,
        help="Image file path or folder path to predict"
    )
    args = parser.parse_args()

    predict_path(args.path)