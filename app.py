import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Cargar el modelo
model = load_model('./fashion_mnist.keras')

# Crear la interfaz de usuario
st.title("Clasificador Fashion MNIST")
st.write("Sube una imagen para clasificarla como una categoría de ropa.")

uploaded_file = st.file_uploader("Sube una imagen en escala de grises 28x28 píxeles")

if uploaded_file is not None:
    # Procesar la imagen
    image = Image.open(uploaded_file).convert('L')
    image = image.resize((28, 28))
    image_array = np.array(image) / 255.0  # Normalizar
    image_array = image_array.reshape(1, 28, 28, 1)  # Ajustar dimensiones para el modelo

    # Mostrar la imagen subida
    st.image(image, caption='Imagen subida', use_column_width=True)

    # Predicción
    prediction = model.predict(image_array)
    classes = ['Camiseta/Top', 'Pantalón', 'Suéter', 'Vestido', 'Abrigo', 'Sandalia', 'Camisa', 'Zapatilla', 'Mochila', 'Bota hasta el tobillo']
    
    # Mostrar la predicción correctamente
    st.write(f"Predicción: {classes[np.argmax(prediction)]}")  
