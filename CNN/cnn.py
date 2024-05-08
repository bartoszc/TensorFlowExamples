import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Funkcja do przewidywania obrazu
def predict_image(image_path):
    # Ładowanie obrazu, jego konwersja do tablicy i przeskalowanie
    img = load_img(image_path, target_size=(224, 224))  # Rozmiar wejściowy dla MobileNetV2
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Załadowanie modelu
    model = MobileNetV2(weights='imagenet')

    # Predykcja klasy obrazu
    predictions = model.predict(img_array)
    result = decode_predictions(predictions, top=3)[0]  # Dekodowanie przewidywań na nazwy klas

    return result

# Ścieżka do pliku obrazu
image_path = 'lion.jpg'

# Wywołanie funkcji i wyświetlenie wyników
results = predict_image(image_path)
for i, (imagenet_id, label, score) in enumerate(results):
    print(f"{i+1}: Label: {label}, Score: {score:.2f}")

# Zamykanie sesji TensorFlow, aby zwolnić zasoby
tf.keras.backend.clear_session()
