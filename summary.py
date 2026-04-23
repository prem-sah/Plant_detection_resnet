import tensorflow as tf

model = tf.keras.models.load_model(r"/Users/sushilkumarpatel/Desktop/plant_detection_prem/Plant_disease_detection_resne/plant_disease_model.keras")

print("Model loaded successfully")
model.summary()

# Accuracy 
loss, acc = model.evaluate(val_data)
print("Validation Accuracy:", acc)
print("Validation Loss:", loss)