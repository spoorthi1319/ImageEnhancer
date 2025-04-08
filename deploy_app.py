import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from io import BytesIO

# Load the models once
@st.cache_resource
def load_models():
    model1 = tf.keras.models.load_model("models/gen_em_3.h5", compile=False)
    model2 = tf.keras.models.load_model("models/gen_e_3.h5", compile=False)

    feature_extractor1 = tf.keras.Model(
        inputs=model1.input, outputs=model1.get_layer("add_4").output
    )
    feature_extractor2 = tf.keras.Model(
        inputs=model2.input, outputs=model2.get_layer("add_4").output
    )
    intermediate_model = tf.keras.Model(inputs=model1.input, outputs=model1.output)
    return feature_extractor1, feature_extractor2, intermediate_model

# Combine models
def combine_srgan_models(feature_extractor1, feature_extractor2, intermediate_model, input_img, alpha=0.5):
    if len(input_img.shape) == 3:
        input_img = np.expand_dims(input_img, axis=0)

    features1 = feature_extractor1.predict(input_img)
    features2 = feature_extractor2.predict(input_img)

    fused_features = alpha * features1 + (1 - alpha) * features2
    output_img = intermediate_model.predict(input_img)
    return output_img

# UI
st.set_page_config(layout="centered")
st.title("🧠 Image Enhancer")
st.write("Upload a low-resolution image, click **Enhance**, and download the enhanced version.")

# Upload section
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Read image and preprocess
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    st.image(image_rgb, caption="Original Image", use_column_width=True)

    # Resize and normalize
    input_img = cv2.resize(image_rgb, (32, 32)).astype(np.float32) / 255.0
    input_img_expanded = np.expand_dims(input_img, axis=0)

    # Load models
    feature_extractor1, feature_extractor2, intermediate_model = load_models()

    # Create two columns for buttons
    col1, col2 = st.columns(2)
    
    with col1:
        # Enhance button
        if st.button("✨ Enhance Image", help="Click to enhance your image"):
            st.write("Enhancing image, please wait...")
            enhanced = combine_srgan_models(feature_extractor1, feature_extractor2, intermediate_model, input_img_expanded)
            enhanced_img = np.clip(enhanced[0], 0, 1)
            enhanced_img_uint8 = (enhanced_img * 255).astype("uint8")

            st.session_state.enhanced_image = enhanced_img_uint8  # Store in session state
            st.image(enhanced_img_uint8, caption="Enhanced Image", use_column_width=True)
            st.success("Image enhancement complete!")

    # Check if enhanced image exists in session state
    if 'enhanced_image' in st.session_state:
        with col2:
            # Download button
            image_pil = Image.fromarray(st.session_state.enhanced_image)
            buf = BytesIO()
            image_pil.save(buf, format="JPEG")
            byte_im = buf.getvalue()

            st.download_button(
                label="📥 Download Enhanced Image",
                data=byte_im,
                file_name="enhanced_image.jpg",
                mime="image/jpeg",
                help="Click to download the enhanced image"
            )