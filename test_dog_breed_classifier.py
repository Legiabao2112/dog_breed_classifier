import os
import json
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Load the trained model and class indices
model = load_model('dog_breed_classifier_model.h5')
print("Model loaded.")

with open('class_indices.json', 'r') as f:
    class_indices = json.load(f)
class_names = {v: k for k, v in class_indices.items()}
print("Class indices loaded.")

# Function to predict breed from an image
def predict_breed(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    pred_index = np.argmax(predictions, axis=1)[0]
    return class_names[pred_index]

# Open a file to store the results
output_file = 'predictions.txt'
with open(output_file, 'w') as f:
    # Loop through test images and save predictions to the file
    test_dir = './data/test'
    for img_file in os.listdir(test_dir):
        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(test_dir, img_file)
            breed = predict_breed(img_path)
            result = f"Image: {img_file}, Predicted Breed: {breed}\n"
            print(result.strip())
            f.write(result)



