# import streamlit as st
# import numpy as np
# from PIL import Image
# import time
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image

# model = load_model('../pnuemonia pred model.keras')

# st.title("📷Pnuemonia Detector")
# st.subheader("Upload an X-Ray image")

# class_labels = ['Not Detected', 'Pneumonia Detected']

# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     img = Image.open(uploaded_file) 

#     st.image(img, caption="Selected Image", width=200)

#     with st.spinner('Detecting...'):
#         img = img.resize((32,32))
#         img_array = image.img_to_array(img)
#         img_array = np.expand_dims(img_array, axis=0)
        
#         prediction = model.predict(img_array)
#         predicted_prob = prediction[0][0]
#         threshold = 0.5  # default decision threshold
#         predicted_label = class_labels[int(predicted_prob > threshold)]
#         confidence = predicted_prob if predicted_prob > threshold else 1 - predicted_prob
#         time.sleep(1) 

#     st.markdown(f"### ✅ Predicted Class: `{predicted_label}`")
#     st.markdown(f"**Confidence:** {confidence:.2f}")


import streamlit as st
import numpy as np
from PIL import Image
import random
import time
import requests
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load model
model = load_model('../pnuemonia pred model.keras')

# Class labels
class_labels = ['Normal', 'Pneumonia']

# Demo image URLs
demo_images = [
    "https://raw.githubusercontent.com/hemlalit/Detecting-Pneumonia-using-CNN/main/demo_images/IM-0001-0001.jpeg",
    "https://raw.githubusercontent.com/hemlalit/Detecting-Pneumonia-using-CNN/main/demo_images/IM-0003-0001.jpeg",
    "https://raw.githubusercontent.com/hemlalit/Detecting-Pneumonia-using-CNN/main/demo_images/IM-0006-0001.jpeg",
    "https://raw.githubusercontent.com/hemlalit/Detecting-Pneumonia-using-CNN/main/demo_images/IM-0011-0001-0001.jpeg"
    "https://raw.githubusercontent.com/hemlalit/Detecting-Pneumonia-using-CNN/main/demo_images/IM-0011-0001-0002.jpeg",
    "https://raw.githubusercontent.com/hemlalit/Detecting-Pneumonia-using-CNN/main/demo_images/person1_virus_13.jpeg",
    "https://raw.githubusercontent.com/hemlalit/Detecting-Pneumonia-using-CNN/main/demo_images/person1_virus_6.jpeg",
    "https://raw.githubusercontent.com/hemlalit/Detecting-Pneumonia-using-CNN/main/demo_images/person1_virus_7.jpeg"
    "https://raw.githubusercontent.com/hemlalit/Detecting-Pneumonia-using-CNN/main/demo_images/person3_virus_15.jpeg",
    "https://raw.githubusercontent.com/hemlalit/Detecting-Pneumonia-using-CNN/main/demo_images/person3_virus_16.jpeg"
]


st.title("📷 Pneumonia Detector")
st.subheader("Upload an X-Ray image or try a demo image")

# Section: Shuffle through demo images
if st.button("🔀 Shuffle & Upload Demo Image"):
    img_url = random.choice(demo_images)
    st.image(img_url, caption="Demo X-ray", width=200)

    with st.spinner("Detecting..."):
        img = Image.open(requests.get(img_url, stream=True).raw)
        img = img.resize((32, 32))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        predicted_prob = prediction[0][0]
        threshold = 0.5
        predicted_label = class_labels[int(predicted_prob > threshold)]
        confidence = predicted_prob if predicted_prob > threshold else 1 - predicted_prob
        time.sleep(1)

    st.markdown(f"### ✅ Predicted Class: `{predicted_label}`")
    st.markdown(f"**Confidence:** {confidence:.2f}")

# Section: Upload file manually
uploaded_file = st.file_uploader("📁 Or upload your own image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", width=200)

    with st.spinner("Detecting..."):
        img = img.resize((32, 32))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        predicted_prob = prediction[0][0]
        threshold = 0.5
        predicted_label = class_labels[int(predicted_prob > threshold)]
        confidence = predicted_prob if predicted_prob > threshold else 1 - predicted_prob
        time.sleep(1)

    st.markdown(f"### ✅ Predicted Class: `{predicted_label}`")
    st.markdown(f"**Confidence:** {confidence:.2f}")
