
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ---------------------------------
# 1. CIFAR-10 class names
# ---------------------------------

class_names = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck'
]


# ---------------------------------
# 2. Load trained model
# ---------------------------------

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
        'best_resnet50_model.keras'
    )
    return model


model = load_model()


# ---------------------------------
# 3. Streamlit interface
# ---------------------------------

st.title("CIFAR-10 Image Classification")

st.write(
    "Upload an image to predict its class "
    "using a ResNet50 deep learning model."
)

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)


# ---------------------------------
# 4. Image preprocessing
# ---------------------------------

def preprocess_image(image):

    # Convert image to RGB
    image = image.convert("RGB")

    # Resize image
    image = image.resize((224, 224))

    # Convert image into NumPy array
    image_array = np.array(image).astype("float32")

    # Scale pixel values to 0-1
    image_array = image_array / 255.0

    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    image_array = (image_array - mean) / std

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    return image_array


# ---------------------------------
# 5. Prediction
# ---------------------------------

if uploaded_file is not None:

    try:
        # Read uploaded image
        image = Image.open(uploaded_file)

        # Display uploaded image
        st.image(
            image,
            caption="Uploaded Image",
            width=300
        )

        if st.button("Predict"):

            # Preprocess image
            processed_image = preprocess_image(image)

            # Model prediction
            predictions = model.predict(
                processed_image,
                verbose=0
            )

            # Get predicted class index
            predicted_index = np.argmax(predictions[0])

            # Get predicted class name
            predicted_class = class_names[predicted_index]

            # Get confidence score
            confidence = predictions[0][predicted_index] * 100

            # Display prediction
            st.subheader("Prediction Result")

            st.success(
                f"Predicted Object: {predicted_class}"
            )

            st.metric(
                label="Confidence Score",
                value=f"{confidence:.2f}%"
            )

    except Exception as e:
        st.error(f"Error: {e}")
