import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import pickle

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model

dataset_path = r"/Users/sushilkumarpatel/Desktop/plant_detection_prem/Plant_disease_detection_resne/archive (2)/PlantVillage"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "plant_disease_model.keras")
HISTORY_PATH = os.path.join(BASE_DIR, "training_history.pkl")
CLASS_MAP_PATH = os.path.join(BASE_DIR, "class_indices.json")

IMG_SIZE = 224
BATCH_SIZE = 32


def build_model(num_classes):

    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(224,224,3)
    )

    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)

    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def train_model():

    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )

    train_data = datagen.flow_from_directory(
        dataset_path,
        target_size=(IMG_SIZE,IMG_SIZE),
        batch_size=BATCH_SIZE,
        subset='training'
    )

    val_data = datagen.flow_from_directory(
        dataset_path,
        target_size=(IMG_SIZE,IMG_SIZE),
        batch_size=BATCH_SIZE,
        subset='validation'
    )

    model = build_model(train_data.num_classes)

    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=10
    )

    # Save model
    model.save(MODEL_PATH)
    print("Model saved:", MODEL_PATH)

    # Save training history
    with open(HISTORY_PATH, "wb") as f:
        pickle.dump(history.history, f)

    print("History saved:", HISTORY_PATH)

    # Save class indices
    with open(CLASS_MAP_PATH, "w") as f:
        json.dump(train_data.class_indices, f)

    print("Class indices saved:", CLASS_MAP_PATH)


if __name__ == "__main__":
    train_model()