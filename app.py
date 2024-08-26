import streamlit as st
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch

# Load the model and processor
processor = AutoImageProcessor.from_pretrained("dima806/facial_emotions_image_detection")
model = AutoModelForImageClassification.from_pretrained("dima806/facial_emotions_image_detection")

# Streamlit app
st.title("Facial Emotion Detection")

# Upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Process the image
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # Get the predicted class
    predicted_class = torch.argmax(outputs.logits, dim=-1).item()

    # Map the predicted class to emotion labels
    emotion_labels = ["sad", "disgust", "angry", "neutral", "fear", "surprise", "happy"]

    if 0 <= predicted_class < len(emotion_labels):
        emotion = emotion_labels[predicted_class]
        st.write(f"Predicted Emotion: **{emotion}**")
    else:
        st.write("Predicted class is out of range. Please check the model output.")
