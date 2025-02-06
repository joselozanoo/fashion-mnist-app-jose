import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Cargar el modelo previamente entrenado
model = load_model('./fashion_mnist.keras')  # Asegúrate de tener la ruta correcta al modelo

# Crear interface de usuario
st.title("Clasificador Fashion MNIST")
st.write("Sube una imagen para clasificarla como una categoría de ropa")

# Subir imagen
uploaded_file = st.file_uploader("Sube una imagen en escala de grises de 28x28 píxeles.", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Abrir la imagen y convertirla a escala de grises
    image = Image.open(uploaded_file).convert('L')  # Convertir a Blanco y Negro
    image = image.resize((28, 28))  # Redimensionar la imagen a 28x28 píxeles
    image_array = np.array(image) / 255.0  # Normalizar la imagen para que los valores estén entre 0 y 1
    image_array = image_array.reshape(1, 28, 28, 1)  # Asegurarse de que tenga la forma correcta para el modelo

    # Mostrar la imagen subida
    st.image(image, caption='Imagen subida', use_column_width=True)

    # Realizar la predicción con el modelo cargado
    prediction = model.predict(image_array)
    classes = ["Camiseta/top", "Pantalón", "Jersey", "Vestido", "Abrigo", "Sandalia", "Camisa", "Zapatilla", "Bolso", "Bota"]

    # Mostrar el resultado de la predicción
    st.write("Predicción: ", classes[np.argmax(prediction)])

    # Mostrar probabilidades
    for i, prob in enumerate(prediction[0]):
        st.write(f"{classes[i]}: {prob:.2%}")

    # Clase con mayor probabilidad
    st.write("Predicción:", classes[np.argmax(prediction)])

