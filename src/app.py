import streamlit as st
from PIL import Image
import io
from predict import predict_disease

st.set_page_config(page_title="Universal Plant Doctor", layout="centered", page_icon="🌿")

st.write("Upload a leaf photo to diagnose 38 different diseases instantly. The system will automatically eliminate background noise for accuracy.")

uploaded_file = st.file_uploader("Choose a leaf image.", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the file as raw bytes
    image_bytes = uploaded_file.read()
    original_image = Image.open(io.BytesIO(image_bytes))

    # Create two columns for the Before & After images
    col1, col2 = st.columns(2)
    with col1:
        st.image(original_image, caption="Original Upload", use_container_width=True)

    with st.spinner('Eliminating background and consulting the neural network..'):
        try:
            #One line of code calls the entire backend!
            segmented_img, results = predict_disease(image_bytes)
            
            with col2:
                st.image(segmented_img, caption="AI View (Segmented)", use_container_width=True)

            # Display
            st.subheader("Report:")
            
            for i, res in enumerate(results):
                disease = res['disease']
                conf = res['confidence']
                
                if i == 0:
                    if conf > 80:
                        st.success(f"**1. {disease}**: {conf}% (High Confidence)")
                    elif conf > 50:
                        st.warning(f"**1. {disease}**: {conf}% (Moderate Confidence)")
                    else:
                        st.error(f"**1. {disease}**: {conf}% (Low Confidence - Try a clearer photo)")
                else:
                    st.write(f"**{i+1}.** {disease}: {conf}%")
                    
        except Exception as e:
            st.error(f"An error occurred: {e}")