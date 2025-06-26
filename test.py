import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load the model
model = load_model("vgg16_lung_cancer_cnn.keras")

# Load and preprocess the image
img = load_img("C:/Users/Toshiba/Desktop/Madhu's Project/Data/test/large.cell.carcinoma/000110.png", target_size=(224, 224))  # your CT scan image
img_array = img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions[0])
confidence = predictions[0][predicted_class]

# Class labels (update based on your training)
class_labels = ['adenocarcinoma', 'large.cell.carcinoma', 'normal', 'squamous.cell.carcinoma']

print(f"Prediction: {class_labels[predicted_class]}")
print(f"Confidence: {confidence:.2f}")
