import streamlit as st
import requests
from PIL import Image
import io

import pandas as pd
from search_logger import SearchLogger

st.set_page_config(page_title="Image Captioning", layout="wide", initial_sidebar_state="collapsed")


# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 1rem;
    }
    .stTitle {
        color: #2E4057;
        font-size: 3rem !important;
        padding-bottom: 1rem;
    }
    .upload-box {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .caption-box {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
        margin-top: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)


# Title
st.title("🖼️ Image Captioning App")
st.markdown("### Upload an image to generate a caption")


# Image upload section
with st.container():
    st.markdown('<div class="upload-box">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Button to generate caption
    if st.button("Generate Caption"):
        with st.spinner("Generating caption..."):
            # Convert the image to bytes for sending to the model
            img_bytes = io.BytesIO()
            image.save(img_bytes, format='PNG')
            img_bytes = img_bytes.getvalue()

            # fix
            # Send the image to your model's API for captioning
            response = requests.post("http://backend:8052/caption", files={"image": img_bytes})

            if response.status_code == 200:
                # Display the generated caption
                caption = response.json().get("caption", "No caption generated.")
                st.markdown('<div class="caption-box">', unsafe_allow_html=True)
                st.markdown(f"### Generated Caption: {caption}")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.error("🚫 Error generating caption. Please try again.")
else:
    st.info("👋 Please upload an image to get started.")