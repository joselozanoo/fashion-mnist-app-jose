import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Cargar el modelo
model = load_model('./fashion_mnist.h5')

# Crear la interfaz de usuario
st.title("🛍️ Clasificador Fashion MNIST")
st.write("Sube una imagen para clasificarla como una categoría de ropa.")

uploaded_file = st.file_uploader("Sube una imagen en escala de grises 28x28 píxeles", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Procesar la imagen
    image = Image.open(uploaded_file).convert('L')  # Convertir a escala de grises
    image = image.resize((28, 28))  # Redimensionar
    image_array = np.array(image) / 255.0  # Normalizar valores entre 0 y 1
    image_array = image_array.reshape(1, 28, 28, 1)  # Ajustar dimensiones para el modelo

    # Mostrar la imagen subida
    st.image(image, caption='Imagen subida', use_column_width=True)

    # Predicción
    prediction = model.predict(image_array)
    predicted_class = np.argmax(prediction)  # Índice de la clase con mayor probabilidad
    probability = np.max(prediction)  # Probabilidad máxima

    classes = ['Camiseta/Top', 'Pantalón', 'Suéter', 'Vestido', 'Abrigo', 
               'Sandalia', 'Camisa', 'Zapatilla', 'Mochila', 'Bota hasta el tobillo']

    # Mostrar Predicción
    st.subheader("📌 Resultado de la Predicción")
    st.write(f"🛒 **Categoría:** {classes[predicted_class]}")
    st.write(f"📊 **Confianza:** {probability:.2%}")

    # Gráfico de probabilidades
    st.subheader("📈 Distribución de probabilidades")
    st.bar_chart(prediction[0])

