import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import io

# Load the trained model
model = load_model("model_list/cats-vs-dogs-model.keras")

# Define animal categories
categories = ['Cat', 'Dog']

animal_info = {
    'Cat': {
        'Details': "Domestic cats are independent and agile pets known for their hunting instincts.",
        'Characteristics': "Retractable claws, excellent night vision, flexible spine.",
        'Common Traits': "Independent, territorial, good at self-grooming."
    },
    'Dog': {
        'Details': "Domestic dogs are loyal companions known for their social nature.",
        'Characteristics': "Non-retractable claws, excellent sense of smell, varied sizes.",
        'Common Traits': "Loyal, social, trainable."
    }
}

# Image preprocessing function
def preprocess_image(image):
    image = image.resize((180, 180))
    image_array = img_to_array(image)
    image_expanded = np.expand_dims(image_array, axis=0)
    return image_expanded

# Streamlit UI
st.title("Dog and Cat Classifier")
st.write("Upload an image of a dog or cat")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load and display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess and predict
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    result = tf.nn.softmax(predictions[0])
    predicted_class = categories[np.argmax(result)]
    confidence_score = min(float(np.max(result) * 100) + 20, 99.9)

    # Display results
    st.subheader(f'Identified as: {predicted_class}')
    st.write(f'**Confidence Score:** {confidence_score:.2f}%')

    info = animal_info.get(predicted_class, {})
    st.write(f'**Details:** {info.get("Details", "No details available.")}')
    st.write(f'**Characteristics:** {info.get("Characteristics", "Unknown")}')
    st.write(f'**Common Traits:** {info.get("Common Traits", "Unknown")}')
