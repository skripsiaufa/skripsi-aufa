import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import tempfile

st.markdown(
    """
    <style>
    .stApp {
        background-color: #D9FO8absF ;  
    </style>
    """,
    unsafe_allow_html=True
)
# Streamlit UI

st.markdown('<div style="display: flex; justify-content: flex-end; margin-top:-70px"><img src="https://media.giphy.com/media/X5PsaxTP6U3h9dUSxd/giphy.gif" alt="GIF" width="100%" style="max-width: 200px; margin-right: 250px;"></div>', unsafe_allow_html=True)
st.markdown('<p style="background-color: #658626; color: white; font-size: 30px; padding: 20px; border-radius: 10px; text-align: center; box-shadow: 0px 6px 8px rgba(0, 0, 0, 0.1);">🌻Bunga Herbal App🌻</p>', unsafe_allow_html=True)
st.markdown('<p style="background-color: #8FB447; color: white; font-size: 20px; padding: 10px; border-radius: 5px; text-align: center; box-shadow: 0px 2px 3px rgba(0, 0, 0, 0.1);">🌼 Tipe Bunga 🌼</p>', unsafe_allow_html=True)
st.image("bismillah.png", use_column_width=True)
# Load the trained model
load_model = tf.keras.models.load_model('MobileNet.h5')  # replace with actual model path

# Assuming 'label_names' is a list of class names
label_names = ['asoka', 'kecubung', 'krokot', 'periwinkle', 'telang', 'zinnia']   # replace with actual class names

# Image size to match the input size of your model
IMG_SIZE = (180, 180)  # replace with your actual image size

def predict_image(image):
    img = Image.fromarray(image)
    img = img.resize(IMG_SIZE)
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    classes = load_model.predict(x, batch_size=10)
    outclass = np.argmax(classes)

    return outclass, classes

st.title("Flower Classification")

# File upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    outclass, classes = predict_image(np.array(image))
    
    for i, label in enumerate(label_names):
        if outclass == i:
            predic = classes[0][i]
            st.write(f'Predict Label: {label}')
            st.write(f'Predict Percentage: {predic*100:.02f}')
            st.write(f'List All Predict:\n{classes}')

# Webcam capture
if st.button('Capture from Webcam'):
    st.write("Opening webcam...")
    
    # Capture from webcam
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if ret:
        # Save the frame to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            temp_filename = tmp_file.name
            cv2.imwrite(temp_filename, frame)
        
        image = Image.open(temp_filename)
        st.image(image, caption='Captured Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")

        outclass, classes = predict_image(np.array(image))
        
        for i, label in enumerate(label_names):
            if outclass == i:
                predic = classes[0][i]
                st.write(f'Predict Label: {label}')
                st.write(f'Predict Percentage: {predic*100:.02f}')
                st.write(f'List All Predict:\n{classes}')

    cap.release()
